from graphviz import Digraph


def create_snpio_clustered_diagram(output_filename: str = "snpio_workflow_clustered"):
    """Creates a SNPio workflow diagram with colorblind-friendly palette and visual clusters using Graphviz.

    Args:
        output_filename (str): The name of the output file without extension. Generates PDF, SVG, and PNG.
    """
    dot = Digraph(comment="SNPio Workflow Clustered")

    # Global graph settings
    dot.attr(rankdir="TB", splines="polyline", nodesep="1.2", ranksep="1.5")
    dot.attr(
        "node",
        style="filled",
        shape="box",
        fontname="Helvetica",
        fontsize="12",
        fixedsize="false",
    )

    # Colorblind-Friendly Colors (Wong 2011 palette)
    color_input = "#56B4E9"  # Sky Blue
    color_filter = "#E69F00"  # Vermillion
    color_analysis = "#009E73"  # Bluish Green
    color_encoding = "#F0E442"  # Yellow
    color_visualization = "#CC79A7"  # Reddish Purple
    color_export = "#999999"  # Gray

    # INPUT Cluster
    with dot.subgraph(name="cluster_input") as c:
        c.attr(label="Input", color=color_input, style="filled", fillcolor="#E0F3F8")
        c.node("Input", "VCF Input (VCFReader)", fillcolor=color_input)

    # FILTERING Cluster
    with dot.subgraph(name="cluster_filtering") as c:
        c.attr(
            label="Filtering (NRemover2)",
            color=color_filter,
            style="filled",
            fillcolor="#FFE5B4",
        )
        c.node("FilterMissing", "Missing Data Filtering", fillcolor=color_filter)
        c.node("FilterAllele", "Allele Frequency Filtering", fillcolor=color_filter)
        c.node(
            "FilterBiallelic", "Biallelic/Singleton Filtering", fillcolor=color_filter
        )
        c.node("FilterThinning", "Thinning / Subset (optional)", fillcolor=color_filter)

    # RESOLVE Cluster
    with dot.subgraph(name="cluster_resolve") as c:
        c.attr(
            label="Resolve Filters",
            color=color_filter,
            style="filled",
            fillcolor="#FFE5B4",
        )
        c.node("ResolveFilters", "Resolve Filters", fillcolor=color_filter)

    # ANALYSIS Cluster
    with dot.subgraph(name="cluster_analysis") as c:
        c.attr(
            label="Population Genetic Analysis",
            color=color_analysis,
            style="filled",
            fillcolor="#CCF5E7",
        )
        c.node("Analysis", "Population Genetic Analysis", fillcolor=color_analysis)
        c.node(
            "SummaryStats", "Summary Statistics Calculation", fillcolor=color_analysis
        )
        c.node("FstNei", "Pairwise Fst and Nei's Distance", fillcolor=color_analysis)
        c.node("PCA", "Principal Component Analysis (PCA)", fillcolor=color_analysis)

    # ENCODING Cluster
    with dot.subgraph(name="cluster_encoding") as c:
        c.attr(
            label="SNP Encoding for Machine Learning",
            color=color_encoding,
            style="filled",
            fillcolor="#FFFFD0",
        )
        c.node("Encoding", "SNP Encoding (GenotypeEncoder)", fillcolor=color_encoding)
        c.node("Encoding012", "012-Encoding", fillcolor=color_encoding)
        c.node("EncodingOneHot", "One-Hot Encoding", fillcolor=color_encoding)
        c.node("EncodingInteger", "Integer Encoding", fillcolor=color_encoding)

    # VISUALIZATION Cluster
    with dot.subgraph(name="cluster_visualization") as c:
        c.attr(
            label="Visualization (Plotting)",
            color=color_visualization,
            style="filled",
            fillcolor="#F2D1E4",
        )
        c.node(
            "Visualization", "Visualization (Plotting)", fillcolor=color_visualization
        )
        c.node("Sankey", "Filtering Sankey Diagram", fillcolor=color_visualization)
        c.node("Missingness", "Missing Data Plots", fillcolor=color_visualization)
        c.node("PCAPlot", "PCA Visualization", fillcolor=color_visualization)

    # EXPORT Cluster
    with dot.subgraph(name="cluster_export") as c:
        c.attr(
            label="Export Outputs",
            color=color_export,
            style="filled",
            fillcolor="#DDDDDD",
        )
        c.node(
            "Export", "Export Outputs (VCF, PHYLIP, STRUCTURE)", fillcolor=color_export
        )

    # Edges (Connections)
    dot.edge("Input", "FilterMissing")
    dot.edge("Input", "FilterAllele")
    dot.edge("Input", "FilterBiallelic")
    dot.edge("Input", "FilterThinning")

    dot.edge("FilterMissing", "ResolveFilters")
    dot.edge("FilterAllele", "ResolveFilters")
    dot.edge("FilterBiallelic", "ResolveFilters")
    dot.edge("FilterThinning", "ResolveFilters")

    dot.edge("ResolveFilters", "Analysis")
    dot.edge("ResolveFilters", "Encoding")
    dot.edge("ResolveFilters", "Visualization")
    dot.edge("ResolveFilters", "Export")

    dot.edge("Analysis", "SummaryStats")
    dot.edge("Analysis", "FstNei")
    dot.edge("Analysis", "PCA")

    dot.edge("Encoding", "Encoding012")
    dot.edge("Encoding", "EncodingOneHot")
    dot.edge("Encoding", "EncodingInteger")

    dot.edge("Visualization", "Sankey")
    dot.edge("Visualization", "Missingness")
    dot.edge("Visualization", "PCAPlot")

    # Output files
    dot.render(filename=output_filename, format="pdf", cleanup=True)
    dot.render(filename=output_filename, format="svg", cleanup=True)
    dot.render(filename=output_filename, format="png", cleanup=True)
    print(
        f"Diagram saved as {output_filename}.pdf, {output_filename}.svg, and {output_filename}.png"
    )


if __name__ == "__main__":
    create_snpio_clustered_diagram("scripts/snpio_workflow_clustered")
