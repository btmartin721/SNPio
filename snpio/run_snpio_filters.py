import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main():
    args = parse_args()
    print("Using snpio version:", get_snpio_version())

    loader = BenchmarkDataLoader(snpio_path=args.snpio_path)

    df_mem = loader.load_memory("nremover", "SNPio")
    df_mem_snpfiltr = loader.load_memory("snpfiltr", "SNPfiltR")

    df_mem_long = melt_metrics(df_mem, "memory_usage")
    df_mem_snpfiltr_long = melt_metrics(df_mem_snpfiltr, "memory_usage")

    df_mem_long = df_mem_long[["run_id", "NLoci", "type", "metric", "value"]]

    df_mem_snpfiltr_long = df_mem_snpfiltr_long[
        ["run_id", "NLoci", "type", "metric", "value"]
    ]

    df_long = pd.concat([df_mem_long, df_mem_snpfiltr_long], ignore_index=True)

    plot_benchmark_results(df_long, args.output_dir, "memory_usage")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark SNPio and vcfR performance."
    )
    parser.add_argument(
        "--repeats", type=int, default=100, help="Number of replicates per test"
    )
    parser.add_argument(
        "--snpio_path", type=Path, required=True, help="Directory with SNPio JSONs"
    )
    parser.add_argument(
        "--output_dir", type=Path, required=True, help="Directory for saving plots"
    )
    return parser.parse_args()


def get_snpio_version():
    import snpio

    return snpio.__version__


class BenchmarkDataLoader:
    def __init__(self, snpio_path: Path):
        self.snpio_path = snpio_path

    def load_runtime(self, method, source_type):
        dflist = []
        path = self.snpio_path
        for f in path.glob(f"{method}_*metrics.json"):  # safe pattern
            with open(f, "r") as fh:
                data = json.load(fh)
            filt_list = []
            for k, v in data.items():
                dftmp = pd.DataFrame(v)
                dftmp["Filter Method"] = k
                dftmp["type"] = "SNPio"
                import sys

                sys.exit()
                filt_list.append(dftmp)
            df = pd.concat(filt_list, ignore_index=True)
            df["NLoci"] = f.stem.split("_")[-2]
            df["run_id"] = range(len(df))
            df["method"] = method
            df["type"] = source_type
            dflist.append(df)
        return pd.concat(dflist, ignore_index=True).rename(
            columns={0: "execution_time"}
        )

    def load_memory(self, method, source_type):
        dflist = []
        path = self.snpio_path

        is_snpio = False
        # Use separate patterns for each method
        if method in {"VCFReader", "nremover"}:
            is_snpio = True
            pattern = f"{method}_memory_usage_*_loci.json"
        elif method in {"vcfR", "snpfiltr"}:
            # This pattern is used for vcfR and snpfiltr
            pattern = f"{method}_memory_usage_*.json"
        else:
            raise ValueError(f"Unsupported method for memory load: {method}")

        for f in path.glob(pattern):
            with open(f, "r") as fh:
                data = json.load(fh)
                filt_list = []
            for k, v in data.items():
                try:
                    dftmp = pd.DataFrame.from_dict(v, orient="index")
                except AttributeError:
                    dftmp = pd.DataFrame(v)
                if is_snpio:
                    dftmp = dftmp.reset_index()
                    dftmp = dftmp.rename(columns={"index": "run_id"})
                else:
                    dftmp["run_id"] = dftmp["run_id"] - 1

                dftmp["Filter Method"] = k
                filt_list.append(dftmp)

            df = pd.concat(filt_list, ignore_index=True)

            if is_snpio:
                # Extract the NLoci from the filename
                df["NLoci"] = f.stem.split("_")[-2]
            else:
                # Extract the NLoci from the filename
                df["NLoci"] = f.stem.split("_")[-1]

            df["method"] = method
            df["type"] = source_type
            dflist.append(df)

        # Concatenate all dataframes in the list into a single dataframe
        # and reset the index
        df_all = pd.concat(dflist, ignore_index=True)

        return df_all.rename(columns={"peak_rss_MB": "memory_usage", 0: "memory_usage"})


def melt_metrics(df, metric):
    df["NLoci"] = df["NLoci"].astype(int)

    df_melt = df.melt(
        id_vars=["method", "run_id", "type", "NLoci", "Filter Method"],
        value_vars=["memory_usage"],
        var_name="metric",
        value_name="value",
    )
    df_melt["NLoci"] = df_melt["NLoci"].astype(int)
    return df_melt.sort_values(by=["NLoci", "Filter Method", "run_id"])


def plot_benchmark_results(df_long, output_dir, metric):
    output_dir.mkdir(parents=True, exist_ok=True)
    df_long["NLoci"] = df_long["NLoci"].astype(int)
    df_long = df_long.sort_values(
        by=["NLoci", "type", "run_id"], ascending=[True, False, True]
    )
    df_long["NLoci"] = df_long["NLoci"].astype(str)

    fontsize = 20
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": fontsize,
            "axes.titlesize": fontsize,
            "axes.labelsize": fontsize,
            "xtick.labelsize": fontsize,
            "ytick.labelsize": fontsize,
        }
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.set_style("white")

    sns.lineplot(
        data=df_long,
        x="NLoci",
        y="value",
        hue="type",
        palette="Set2",
        legend=True,
        err_style="bars",
        marker="o",
        lw=3,
        markersize=10,
        errorbar="sd",
        err_kws={"capsize": 15},
        ax=ax,
    )

    ylabel = "Execution Time (s)" if metric == "execution_time" else "Memory Usage (MB)"
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Number of Loci")

    for fmt in ["png", "pdf"]:
        fig.savefig(
            output_dir / f"nremover2_{metric}_summary.{fmt}",
            dpi=300,
            bbox_inches="tight",
        )


if __name__ == "__main__":
    main()
