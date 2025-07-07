import importlib
import logging
from pathlib import Path
from typing import Any, Dict, List

import multiqc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
from multiqc import BaseMultiqcModule
from multiqc import config as mqc_config
from multiqc.plots import bargraph, box, heatmap, linegraph, scatter, table, violin
from scipy.stats import gaussian_kde

from snpio.utils.plot_queue import queued_plots

LOG = logging.getLogger("snpio.multiqc")


def custom_linegraph_kde_plot(
    zscores: pd.Series,
    *,
    title: str = "Distribution of Observed Z-Scores for D-Statistics",
    xlabel: str = "Z-Score",
    ylabel: str = "Estimated Density",
    kde_bw: float = None,
    x_range: tuple[float, float] = (-3.5, 4.5),
    resolution: int = 500,
) -> str:
    """
    Creates a smoothed KDE line plot of Z-scores using Plotly Express.

    Args:
        zscores (pd.Series): 1D series of Z-score values.
        title (str): Plot title.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        kde_bw (float, optional): Bandwidth override for KDE.
        x_range (tuple): Range of x-axis (Z-score values).
        resolution (int): Number of points to evaluate KDE.

    Returns:
        str: HTML div of the Plotly plot (can be returned by custom_linegraph).
    """
    # Drop missing values
    z = zscores.dropna().values

    # Perform KDE
    kde = gaussian_kde(z, bw_method=kde_bw)
    x_vals = np.linspace(x_range[0], x_range[1], resolution)
    y_vals = kde(x_vals)

    # Create DataFrame for Plotly
    df = pd.DataFrame({"Z-Score": x_vals, "Density": y_vals})

    # Create the plot
    fig = px.line(
        df,
        x="Z-Score",
        y="Density",
        title=title,
    )
    fig.update_layout(
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        template="plotly_white",
        hovermode="x unified",
    )

    # Return the HTML string (for MultiQC custom report section)
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def custom_box(plotdata: pd.DataFrame, pconfig: dict) -> str:
    """Generate a custom Plotly box plot with locus-level tooltips for MultiQC.

    Args:
        plotdata (pd.DataFrame): DataFrame with summary statistics (Ho, He, Pi) as columns and loci as index.
        pconfig (dict): Not used directly, but passed by MultiQC.

    Returns:
        str: HTML string for rendering in MultiQC report.
    """
    # Reset index so "Locus (CHROM:POS)" becomes a column
    df = plotdata.reset_index().melt(
        id_vars=["Locus (CHROM:POS)"], var_name="Statistic", value_name="Value"
    )

    points = pconfig.get("points", "outliers")
    if points not in {"all", "outliers"}:
        LOG.warning(
            f"Invalid points value '{points}' in pconfig. Defaulting to 'outliers'."
        )
        points = "outliers"

    # Create a box plot with overlaid points (jittered)
    fig = px.box(
        df,
        x="Value",
        y="Statistic",
        orientation="h",
        points=points,  # shows only outlier points over the boxes
        hover_data={"Locus (CHROM:POS)": True, "Value": True},
        title="Summary Statistics per Locus",
    )

    # Optional: customize layout
    fig.update_traces(jitter=0.5, marker=dict(size=4))
    fig.update_layout(
        yaxis_title="Summary Statistic",
        xaxis_title="Value",
        hoverlabel=dict(bgcolor="white", font_size=12),
        margin=dict(t=40, b=40, l=40, r=40),
        height=500,
        template="plotly_white",
    )

    return pio.to_html(fig, include_plotlyjs="cdn", full_html=False)


import numpy as np
import pandas as pd
import plotly.express as px
from scipy.stats import gaussian_kde


def custom_linegraph_kde_plot(
    zscores: pd.Series,
    *,
    title: str = "Distribution of Observed Z-Scores for D-Statistics",
    xlabel: str = "Z-Score",
    ylabel: str = "Estimated Density",
    kde_bw: float = None,
    x_range: tuple[float, float] = (-3.5, 4.5),
    resolution: int = 500,
) -> str:
    """
    Creates a smoothed KDE line plot of Z-scores using Plotly Express.

    Args:
        zscores (pd.Series): 1D series of Z-score values.
        title (str): Plot title.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        kde_bw (float, optional): Bandwidth override for KDE.
        x_range (tuple): Range of x-axis (Z-score values).
        resolution (int): Number of points to evaluate KDE.

    Returns:
        str: HTML div of the Plotly plot (can be returned by custom_linegraph).
    """
    # Drop missing values
    z = zscores.dropna().values

    # Perform KDE
    kde = gaussian_kde(z, bw_method=kde_bw)
    x_vals = np.linspace(x_range[0], x_range[1], resolution)
    y_vals = kde(x_vals)

    # Create DataFrame for Plotly
    df = pd.DataFrame({"Z-Score": x_vals, "Density": y_vals})

    # Create the plot
    fig = px.line(df, x="Z-Score", y="Density", title=title)
    fig.update_layout(
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        template="simple_white",
        hovermode="x unified",
    )

    # Return the HTML string (for MultiQC custom report section)
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def _build_snpio_report(
    *,
    prefix: str | Path,
    output_dir: str | Path | None = None,
    title: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Build a MultiQC report for a SNPio run.

    This function initializes the SNPioMultiQC module, resolves core locations, patches MultiQC's configuration with a custom logo, and writes the report to the specified output directory.

    Args:
        prefix: The path to the input files (e.g. VCF, BCF).
        output_dir: The directory where the report will be saved.
        title: The title of the report.
        overwrite: Whether to overwrite existing reports.

    Returns:
        Path: The path to the generated MultiQC report.

    Raises:
        FileNotFoundError: If the custom logo is not found.
        ValueError: If the output directory is not specified.
    """
    # ------------------------------------------------------------------ #
    # 1. Resolve core locations                                          #
    # ------------------------------------------------------------------ #
    prefix = Path(prefix).expanduser().resolve()
    report_dir = (
        Path(output_dir).expanduser()
        if output_dir
        else prefix.parent / f"{prefix.name}_output" / "multiqc"
    )
    report_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 2. Make the logo path absolute and patch MultiQC’s config          #
    # ------------------------------------------------------------------ #
    pkg_root = Path(importlib.import_module("snpio").__file__).parent
    logo_path = (pkg_root / "img" / "snpio_logo.png").resolve()

    if not logo_path.exists() or not logo_path.is_file():
        raise FileNotFoundError(f"Custom logo not found: {logo_path}")

    # Update the in-memory MultiQC config so every downstream call sees it
    mqc_config.update({"custom_logo": str(logo_path)})

    # (Optional) — if I ever want to emit a YAML file for debugging
    # with open(report_dir / "multiqc_config.yaml", "w") as f:
    #     yaml.safe_dump({"custom_logo": str(logo_path)}, f)

    # ------------------------------------------------------------------ #
    # 3. Register the SNPio module and write the report                  #
    # ------------------------------------------------------------------ #
    module = SNPioMultiQC()  # triggers __init__ & parse_logs()
    multiqc.report.modules.append(module)

    multiqc.write_report(
        output_dir=report_dir,
        title=title or f"SNPio MultiQC Report - {prefix.name}",
        force=overwrite,
        filename="multiqc_report.html",
        make_data_dir=True,
    )

    html = report_dir / "multiqc_report.html"
    LOG.info(f"MultiQC report written → {html}")
    return html


# --------------------------------------------------------------------------- #
# 1.                   SNPio MultiQC module                                 #
# --------------------------------------------------------------------------- #
class SNPioMultiQC(BaseMultiqcModule):
    """A MultiQC module and an in-process builder.

    This module is designed to queue plots generated by the SNPio pipeline and render them in a MultiQC report. It provides class methods to queue different types of plots (tables, heatmaps, barplots) and automatically renders them when the module is initialized or when `parse_logs()` is called.
    """

    # -----------------------  module constructor  --------------------------- #
    def __init__(self) -> None:
        """Initialize the SNPioMultiQC module.

        This module is designed to queue plots generated by the SNPio pipeline and render them in a MultiQC report. It provides class methods to queue different types of plots (tables, heatmaps, barplots) and automatically renders them when the module is initialized or when `parse_logs()` is called.
        """

        super().__init__(
            name="SNPio",
            anchor="snpio",
            target="SNPio",
            href="https://github.com/btmartin721/SNPio",
            info="Plots generated by the SNPio pipeline.",
        )

        self.parse_logs()

    # .....................  public “queue” helpers  ........................ #
    @classmethod
    def queue_table(
        cls,
        df: pd.DataFrame | pd.Series,
        *,
        panel_id: str,
        section: str,
        title: str,
        index_label: str,
        description: str | None = None,
        pconfig: Dict[str, Any] | None = None,
        headers: Dict[str, Any] | None = None,
    ) -> None:
        """Queue a table for rendering in the MultiQC report.

        This method converts a DataFrame into a format suitable for MultiQC and queues it for rendering.

        Args:
            df (pd.DataFrame): DataFrame containing the data to plot.
            panel_id (str): Unique identifier for the plot panel.
            section (str): Section name in the MultiQC report.
            title (str): Title of the plot.
            index_label (str): Label for the index column.
            description (str, optional): Description text for the plot. Defaults to None.
            pconfig (Dict[str, Any], optional): Additional configuration for the plot. Defaults to
                None.
            headers (Dict[str, Any], optional): Headers for the table. If provided, it should contain keys like 'max', 'min', 'scale', and 'suffix' to customize the table  headers. Defaults to None.
        """
        df = cls._series_to_dataframe(df, index_label, "value")
        data = cls._df_to_plot_data(df, index_label=index_label)

        d = {
            "kind": "table",
            "data": data,
            "panel_id": panel_id,
            "section": section,
            "title": title,
            "index_label": index_label,
            "description": description or "",
            "pconfig": pconfig,
        }

        if headers is not None:
            d["headers"] = headers

        queued_plots.append(d)

    @classmethod
    def queue_html(
        cls,
        html: str | Path,
        *,
        panel_id: str,
        section: str,
        title: str,
        index_label: str,
        description: str | None = None,
    ) -> None:
        """Queue an HTML snippet for rendering in the MultiQC report.

        This method queues an HTML snippet for rendering in the MultiQC report.

        Args:
            cls: The class instance.
            html (str | Path): The HTML snippet or file path to queue.
            panel_id (str): Unique identifier for the plot panel.
            section (str): Section name in the MultiQC report.
            title (str): Title of the plot.
            index_label (str): Label for the index column.
            description (str, optional): Description text for the plot. Defaults to None.
        """
        queued_plots.append(
            {
                "kind": "html",
                "data": html,
                "panel_id": panel_id,
                "section": section,
                "title": title,
                "index_label": index_label,
                "description": description or "",
            }
        )

    @classmethod
    def queue_heatmap(
        cls,
        df: pd.DataFrame | pd.Series,
        *,
        panel_id: str,
        section: str,
        title: str,
        index_label: str,
        description: str | None = None,
        pconfig: Dict[str, Any] | None = None,
    ) -> None:
        """Queue a heatmap for rendering in the MultiQC report.

        This method converts a DataFrame into a format suitable for MultiQC heatmaps and queues it for rendering.

        Args:
            df (pd.DataFrame): DataFrame containing the data to plot.
            panel_id (str): Unique identifier for the plot panel.
            section (str): Section name in the MultiQC report.
            title (str): Title of the plot.
            index_label (str): Label for the index column.
            description (str, optional): Description text for the plot. Defaults to None.
            pconfig (Dict[str, Any], optional): Additional configuration for the plot. Defaults to None.
        """

        df = cls._series_to_dataframe(df, index_label=index_label, value_label="value")
        data = cls._df_to_plot_data(df, index_label=index_label)

        queued_plots.append(
            {
                "kind": "heatmap",
                "data": data,
                "panel_id": panel_id,
                "section": section,
                "title": title,
                "index_label": index_label,
                "description": description or "",
                "pconfig": pconfig
                or {"id": panel_id, "title": f"SNPio: {title} Heatmap"},
            }
        )

    @classmethod
    def queue_barplot(
        cls,
        df: pd.Series | pd.DataFrame,
        *,
        panel_id: str,
        section: str,
        title: str,
        index_label: str,
        value_label: str = "value",
        description: str | None = None,
        pconfig: Dict[str, Any] | None = None,
        cats: List[str] | Dict[str, Dict[str, str]] | None = None,
    ) -> None:
        """Queue a bar plot for rendering in the MultiQC report.

        This method converts a DataFrame or Series into a format suitable for MultiQC bar plots and queues it for rendering.

        Args:
            df (pd.Series | pd.DataFrame): DataFrame or Series containing the data to plot.
            panel_id (str): Unique identifier for the plot panel.
            section (str): Section name in the MultiQC report.
            title (str): Title of the plot.
            index_label (str): Label for the index column.
            value_label (str, optional): Label for the value column. Defaults to "value".
            description (str, optional): Description text for the plot. Defaults to None.
            pconfig (Dict[str, Any], optional): Additional configuration for the plot. Defaults to None.
            cats (List[str] | Dict[Dict[str, str]], optional): Categories for the plot. Supported inner keys (if dict) are 'name' and 'color'. Defaults to None.
        """
        if isinstance(df, list):
            data = []
            for d in df:
                dftmp = cls._series_to_dataframe(d, index_label, value_label)
                d = cls._df_to_plot_data(dftmp, index_label=index_label)
                data.append(d)
        else:
            df = cls._series_to_dataframe(df, index_label, value_label)
            data = cls._df_to_plot_data(df, index_label=index_label)

        d = {
            "kind": "bar",
            "data": data,
            "panel_id": panel_id,
            "section": section,
            "title": title,
            "index_label": index_label,
            "value_label": value_label,
            "description": description or "",
            "pconfig": pconfig or {"id": panel_id, "title": f"SNPio: {title} Barplot"},
        }

        if cats is not None:
            d["cats"] = cats

        queued_plots.append(d)

    @classmethod
    def queue_violin(
        cls,
        df: pd.DataFrame | pd.Series,
        *,
        panel_id: str,
        section: str,
        title: str,
        index_label: str,
        description: str | None = None,
        pconfig: Dict[str, Any] | None = None,
    ) -> None:
        """Queue a violin plot for rendering in the MultiQC report.

        Args:
            df (pd.DataFrame | pd.Series): DataFrame or Series containing the data to plot.
            panel_id (str): Unique identifier for the plot panel.
            section (str): Section name in the MultiQC report.
            title (str): Title of the plot.
            index_label (str): Label for the index column.
            description (str, optional): Description text for the plot. Defaults to None.
            pconfig (Dict[str, Any], optional): Additional configuration for the plot. Defaults to None.
        """
        df = cls._series_to_dataframe(df, index_label, "value")
        data = cls._df_to_plot_data(df, index_label=index_label)

        queued_plots.append(
            {
                "kind": "violin",
                "data": data,
                "panel_id": panel_id,
                "section": section,
                "title": title,
                "index_label": index_label,
                "description": description or "",
                "pconfig": pconfig
                or {"id": panel_id, "title": f"SNPio: {title} Violin Plot"},
            }
        )

    @classmethod
    def queue_custom_lineplot(
        cls,
        df: pd.DataFrame | pd.Series,
        *,
        panel_id: str,
        section: str,
        title: str,
        index_label: str,
        description: str | None = None,
        pconfig: Dict[str, Any] | None = None,
    ) -> None:
        """Queue a custom line plot for rendering in the MultiQC report."""

        if not pd.isnull(df).all():
            queued_plots.append(
                {
                    "kind": "custom_line",
                    "data": df,
                    "panel_id": panel_id,
                    "section": section,
                    "title": title,
                    "index_label": index_label,
                    "description": description or "",
                    "pconfig": pconfig
                    or {"id": panel_id, "title": f"SNPio: {title} Custom Line Plot"},
                }
            )
        else:
            LOG.warning(
                f"Skipping custom line plot for {panel_id} in section {section} due to all NaN values."
            )

    @classmethod
    def queue_custom_boxplot(
        cls,
        df: pd.DataFrame | pd.Series,
        *,
        panel_id: str,
        section: str,
        title: str,
        index_label: str,
        description: str | None = None,
        pconfig: Dict[str, Any] | None = None,
    ) -> None:
        """Queue a custom box plot for rendering in the MultiQC report."""
        queued_plots.append(
            {
                "kind": "custom_box",
                "data": df,
                "panel_id": panel_id,
                "section": section,
                "title": title,
                "index_label": index_label,
                "description": description or "",
                "pconfig": pconfig
                or {"id": panel_id, "title": f"SNPio: {title} Custom Boxplot"},
            }
        )

    @classmethod
    def queue_boxplot(
        cls,
        df: pd.DataFrame | pd.Series,
        *,
        panel_id: str,
        section: str,
        title: str,
        index_label: str,
        description: str | None = None,
        pconfig: Dict[str, Any] | None = None,
    ) -> None:
        """Queue a box plot for rendering in the MultiQC report.

        Args:
            df (pd.DataFrame | pd.Series): DataFrame or Series containing the data to plot.
            panel_id (str): Unique identifier for the plot panel.
            section (str): Section name in the MultiQC report.
            title (str): Title of the plot.
            index_label (str): Label for the index column.
            description (str, optional): Description text for the plot. Defaults to None.
            pconfig (Dict[str, Any], optional): Additional configuration for the plot. Defaults to None.
        """
        try:
            df = cls._series_to_dataframe(df, index_label, "value")
            data = cls._df_to_plot_data(df, index_label=index_label)
        except TypeError:
            data = df

        queued_plots.append(
            {
                "kind": "box",
                "data": data,
                "panel_id": panel_id,
                "section": section,
                "title": title,
                "index_label": index_label,
                "description": description or "",
                "pconfig": pconfig
                or {"id": panel_id, "title": f"SNPio: {title} Boxplot"},
            }
        )

    @classmethod
    def queue_linegraph(
        cls,
        data: Dict[str, Dict[int, int]],
        *,
        panel_id: str,
        section: str,
        title: str,
        index_label: str,
        description: str | None = None,
        pconfig: Dict[str, Any] | None = None,
    ) -> None:
        """Queue a line graph for rendering in the MultiQC report.

        Args:
            data (Dict[str, Dict[int, int]]): Data to plot, where keys are sample names and values are dictionaries with x-axis values and their corresponding y-axis values.
            panel_id (str): Unique identifier for the plot panel.
            section (str): Section name in the MultiQC report.
            title (str): Title of the plot.
            index_label (str): Label for the index column.
            description (str, optional): Description text for the plot. Defaults to None.
            pconfig (Dict[str, Any], optional): Additional configuration for the plot. Defaults to None.
        """
        queued_plots.append(
            {
                "kind": "linegraph",
                "data": data,
                "panel_id": panel_id,
                "section": section,
                "title": title,
                "index_label": index_label,
                "description": description or "",
                "pconfig": pconfig or {"id": panel_id, "title": f"{title} Line Graph"},
            }
        )

    @classmethod
    def queue_scatterplot(
        cls,
        df: pd.DataFrame | pd.Series,
        *,
        panel_id: str,
        section: str,
        title: str,
        index_label: str,
        description: str | None = None,
        pconfig: Dict[str, Any] | None = None,
    ) -> None:
        """Queue a scatter plot for rendering in the MultiQC report.

        Args:
            df (pd.DataFrame | pd.Series): DataFrame or Series containing the data to plot.
            panel_id (str): Unique identifier for the plot panel.
            section (str): Section name in the MultiQC report.
            title (str): Title of the plot.
            index_label (str): Label for the index column.
            description (str, optional): Description text for the plot. Defaults to None.
            pconfig (Dict[str, Any], optional): Additional configuration for the plot. Defaults to None.
        """
        df = cls._series_to_dataframe(df, index_label, "value")
        data = cls._df_to_plot_data(df, index_label=index_label)

        queued_plots.append(
            {
                "kind": "scatter",
                "data": data,
                "panel_id": panel_id,
                "section": section,
                "title": title,
                "index_label": index_label,
                "description": description or "",
                "pconfig": pconfig
                or {"id": panel_id, "title": f"SNPio: {title} Scatterplot"},
            }
        )

    @classmethod
    def _series_to_dataframe(cls, df, index_label, value_label):
        """Convert a Series to a DataFrame with specified index and value labels.

        This method converts a Series or DataFrame into a DataFrame with specified index and value labels.

        Args:
            df (pd.Series | pd.DataFrame): Series or DataFrame to convert.
            index_label (str): Label for the index column.
            value_label (str): Label for the value column.

        Returns:
            pd.DataFrame: DataFrame with the specified index and value labels.
        """
        if isinstance(df, pd.Series):
            df = df.to_frame(name=value_label).reset_index()
            df.columns = [index_label, value_label]
        elif isinstance(df, pd.DataFrame):
            df = df.copy()
            if df.index.name != index_label:
                df.index.name = index_label
        elif isinstance(df, dict):
            df = df.copy()
        else:
            raise TypeError(
                f"Unsupported type for df: {type(df)}. Expected pd.Series, pd.DataFrame, or dict."
            )

        return df

    # ......................... Render dispatcher ..........................
    def parse_logs(self) -> None:
        """Render the queued plots in the MultiQC report.

        This method processes the queued plots and renders them in the MultiQC report. It iterates through the `queued_plots` list, checking the `kind` of each entry and calling the appropriate rendering method. After processing, it clears the queue to prevent re-rendering.
        """
        if not queued_plots:
            LOG.warning("No plots queued for rendering.")
            return

        LOG.info(f"Rendering {len(queued_plots)} queued plots in MultiQC report...")

        try:
            for entry in queued_plots:
                kind = entry["kind"].lower()
                if kind == "table":
                    plot_obj = self._add_table(entry)
                elif kind == "heatmap":
                    plot_obj = self._add_heatmap(entry)
                elif kind == "bar":
                    plot_obj = self._add_barplot(entry)
                elif kind == "scatter":
                    plot_obj = self._add_scatterplot(entry)
                elif kind == "violin":
                    plot_obj = self._add_violinplot(entry)
                elif kind == "box":
                    plot_obj = self._add_boxplot(entry)
                elif kind == "custom_box":
                    plot_obj = self._add_custom_boxplot(entry)
                elif kind == "linegraph":
                    plot_obj = self._add_linegraph(entry)
                elif kind == "custom_line":
                    plot_obj = self._add_custom_lineplot(entry)
                elif kind == "html":
                    d = self._add_html(entry)
                else:
                    LOG.warning(f"Unknown plot kind: {kind}")

                if kind == "html":
                    plot_obj = None
                    content = d["html_content"]

                    self.add_section(
                        plot=plot_obj,
                        name=entry["title"],
                        anchor=entry["panel_id"],
                        description=entry.get("description", ""),
                        helptext=entry.get("helptext", ""),
                        content=content,
                    )

                else:
                    self.add_section(
                        name=entry["title"],
                        anchor=entry["panel_id"],
                        description=entry.get("description", ""),
                        helptext=entry.get("helptext", ""),
                        plot=plot_obj,
                    )

        except Exception as e:
            LOG.error(f"Failed to render plot: '{entry['panel_id']}': {e}")
            raise

        # Clear the queue after processing
        queued_plots.clear()

    # .........................  Plot rendering methods  ......................
    ###########################################################################
    def _add_custom_lineplot(self, p: Dict[str, Any]) -> str:
        """Render a custom line plot in the MultiQC report.

        Args:
            p (Dict[str, Any]): Plot parameters including data and metadata.
        """
        zscores = p["data"]
        title = p.get("title", "Custom Line Plot")
        xlabel = p.get("xlabel", "Z-Score")
        ylabel = p.get("ylabel", "Estimated Density")
        kde_bw = p.get("kde_bw", None)
        x_range = p.get("x_range", (-4.5, 4.5))
        resolution = p.get("resolution", 500)

        return custom_linegraph_kde_plot(
            zscores=zscores,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            kde_bw=kde_bw,
            x_range=x_range,
            resolution=resolution,
        )

    def _add_html(self, p: Dict[str, Any]) -> Dict[str, Any | None]:
        """Render an HTML snippet in the MultiQC report.

        Args:
            p (Dict[str, Any]): Plot parameters including data and metadata.
        """
        html: str | Path = p["data"]
        if isinstance(html, (str, Path)) and Path(html).exists():
            with open(html, "r") as f:
                html_content = f.read()
        else:
            msg = "Invalid HTML path."
            self.logger.error(msg)
            raise IOError(msg)

        return {"html_content": html_content, "plot": None}

    def _add_table(self, p: Dict[str, Any]) -> None:
        """Render a table plot in the MultiQC report.

        Args:
            p (Dict[str, Any]): Plot parameters including data and metadata.
        """
        data = p["data"]
        return table.plot(
            data=data, headers=data.get("headers", None), pconfig=p.get("pconfig", None)
        )

    def _add_heatmap(self, p: Dict[str, Any]) -> None:
        """Render a heatmap plot in the MultiQC report.

        Args:
            p (Dict[str, Any]): Plot parameters including data and metadata.
        """

        data = p["data"]
        return heatmap.plot(
            data=data,
            xcats=p.get("xcats", None),
            ycats=p.get("ycats", None),
            pconfig=p.get("pconfig", None),
        )

    def _add_barplot(self, p: Dict[str, Any]) -> None:
        """Render a bar plot in the MultiQC report.

        Args:
            p (Dict[str, Any]): Plot parameters including data and metadata.
        """
        data = p["data"]

        return bargraph.plot(
            data=data, cats=p.get("cats", None), pconfig=p.get("pconfig", None)
        )

    def _add_scatterplot(self, p: Dict[str, Any]) -> None:
        """Render a scatter plot in the MultiQC report.

        Args:
            p (Dict[str, Any]): Plot parameters including data and metadata.
        """
        data = p["data"]
        return scatter.plot(data=data, pconfig=p.get("pconfig", None))

    def _add_violinplot(self, p: Dict[str, Any]) -> None:
        """Render a violin plot in the MultiQC report.

        Args:
            p (Dict[str, Any]): Plot parameters including data and metadata.
        """
        data = p["data"]
        return violin.plot(
            data=data, headers=p.get("headers", None), pconfig=p.get("pconfig", None)
        )

    def _add_boxplot(self, p: pd.DataFrame | pd.Series) -> None:
        """Render a box plot in the MultiQC report.

        Args:
            p (pd.DataFrame | pd.Series): Plot parameters including data and metadata.
        """
        data = p["data"]
        return box.plot(list_of_data_by_sample=data, pconfig=p.get("pconfig", None))

    def _add_custom_boxplot(self, p: pd.DataFrame | pd.Series) -> None:
        """Render a custom box plot in the MultiQC report.

        Args:
            p (pd.DataFrame | pd.Series): Plot parameters including data and metadata.
        """
        data = p["data"]
        return custom_box(plotdata=data, pconfig=p.get("pconfig", None))

    def _add_linegraph(self, p: pd.DataFrame | pd.Series) -> None:
        """Render a line graph in the MultiQC report.

        Args:
            p (pd.DataFrame | pd.Series): Plot parameters including data and metadata.
        """
        data = p["data"]
        return linegraph.plot(data=data, pconfig=p.get("pconfig", None))

    @staticmethod
    def _df_to_plot_data(
        df: pd.DataFrame | Dict[str, Any], index_label: str | None = None
    ) -> Dict[str, Any]:
        """Convert a DataFrame to a dictionary suitable for MultiQC plots.

        This method converts a DataFrame into a dictionary with 'headers' and 'data' keys, where 'headers' contains the column names and 'data' contains the rows as dictionaries. If an `index_label` is provided, the index will be reset and added as a named column.

        Args:
            df (pd.DataFrame | Dict[str, Any]): The DataFrame to convert.
            index_label (str | None): Optional name for the index column. If provided, the index will be reset and added as a named column.

        Returns:
            Dict[str, Any]: Dictionary with 'data' key for use in plots.

        Example:
            >>> df = pd.DataFrame({
            ...     "missing": [0.1, 0.2],
            ...     "heterozygosity": [0.3, 0.4]
            ... }, index=["sample1", "sample2"])
            >>> SNPioMultiQC._df_to_plot_data(df, index_label="Sample")
            {
                'headers': ['Sample', 'missing', 'heterozygosity'],
                'data': [
                    {'Sample': 'sample1', 'missing': 0.1, 'heterozygosity': 0.3},
                    {'Sample': 'sample2', 'missing': 0.2, 'heterozygosity': 0.4}
                ]
            }
        """
        if isinstance(df, dict):
            return df

        df = df.copy()
        if index_label is not None:
            if df.index.name is not None and df.index.name != index_label:
                df.index.name = index_label
            else:
                if index_label in df.columns:
                    df = df.set_index(index_label)
                else:
                    df.index.name = index_label

        return df.to_dict(orient="index")

    def build_report(self, **kwargs):
        """Instance-friendly alias to write the MultiQC report to an HTML file.

        This method is a convenience wrapper around the `_build_snpio_report` function, allowing you to generate a MultiQC report using the queued plots.

        Args:
            prefix: The prefix for the report files.
            output_dir: Directory to write the report to (default: `<prefix>_output/multiqc`).
            title: Title for the report (default: `SNPio QC Report - <prefix>`).
            overwrite: Whether to overwrite existing report files (default: `False`).

        Returns:
            Path to the generated MultiQC report HTML file.

        Example:
            >>> from snpio import SNPioMultiQC
            >>> # Queue some plots first
            >>> mqc = SNPioMultiQC()
            >>> html_file = mqc.build_report(
            ...     prefix="run1",
            ...     output_dir="/path/to/output",
            ...     title="My SNPio Report",
            ...     overwrite=True
            ... )
        """
        return _build_snpio_report(**kwargs)

    @classmethod
    def build(cls, **kwargs):
        """Class-level alias to write the MultiQC report to an HTML file.

        This method allows you to build the MultiQC report without needing to instantiate the `SNPioMultiQC` class. It is useful for quick report generation in scripts or when you don't need to maintain state between multiple reports.

        Args:
            prefix: The prefix for the report files.
            output_dir: Directory to write the report to (default: `<prefix>_output/multiqc`).
            title: Title for the report (default: `SNPio QC Report - <prefix>`).
            overwrite: Whether to overwrite existing report files (default: `False`).

        Returns:
            Path to the generated MultiQC report HTML file.

        Example:
            >>> from snpio import SNPioMultiQC
            >>> # Queue some plots first
            >>> html_file = SNPioMultiQC.build(
            ...     prefix="run1",
            ...     output_dir="/path/to/output",
            ...     title="My SNPio Report",
            ...     overwrite=True
            ... )

        """
        return _build_snpio_report(**kwargs)
