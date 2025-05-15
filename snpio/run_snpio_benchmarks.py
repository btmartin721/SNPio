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

    loader = BenchmarkDataLoader(snpio_path=args.snpio_path, vcfr_path=args.vcfr_path)

    df_snpio = loader.load_runtime("VCFReader", "SNPio")
    df_snpio_mem = loader.load_memory("VCFReader", "SNPio")
    df_vcfr = loader.load_runtime("vcfR", "vcfR")
    df_vcfr_mem = loader.load_memory("vcfR", "vcfR")

    df_time = pd.concat([df_snpio, df_vcfr], ignore_index=True)
    df_mem = pd.concat([df_snpio_mem, df_vcfr_mem], ignore_index=True)

    df_time_long = melt_metrics(df_time, "execution_time")
    df_mem_long = melt_metrics(df_mem, "memory_usage")

    plot_benchmark_results(df_time_long, args.output_dir, "execution_time")
    plot_benchmark_results(df_mem_long, args.output_dir, "memory_usage")


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
        "--vcfr_path", type=Path, required=True, help="Directory with vcfR JSONs"
    )
    parser.add_argument(
        "--output_dir", type=Path, required=True, help="Directory for saving plots"
    )
    return parser.parse_args()


def get_snpio_version():
    import snpio

    return snpio.__version__


class BenchmarkDataLoader:
    def __init__(self, snpio_path: Path, vcfr_path: Path):
        self.snpio_path = snpio_path
        self.vcfr_path = vcfr_path

    def load_runtime(self, method, source_type):
        dflist = []
        path = self.snpio_path if source_type == "SNPio" else self.vcfr_path
        for f in path.glob(f"{method}_*metrics.json"):  # safe pattern
            with open(f, "r") as fh:
                data = json.load(fh)
            try:
                df = pd.DataFrame(data["results"][0]["times"])
            except KeyError:
                df = pd.DataFrame.from_dict(data, orient="index")
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
        path = self.snpio_path if source_type == "SNPio" else self.vcfr_path

        # Use separate patterns for each method
        if method == "VCFReader":
            pattern = f"{method}_memory_usage_*_loci.json"
        elif method == "vcfR":
            pattern = f"{method}_memory_usage_*.json"
        else:
            raise ValueError(f"Unsupported method for memory load: {method}")

        for f in path.glob(pattern):
            with open(f, "r") as fh:
                data = json.load(fh)
            try:
                df = pd.DataFrame(data["memory_usage"])
                df["NLoci"] = f.stem.split("_")[-1]
            except KeyError:
                df = pd.DataFrame.from_dict(data, orient="index")
                df["NLoci"] = f.stem.split("_")[-2]
            df["method"] = method
            df["type"] = source_type
            dflist.append(df)
        df_all = pd.concat(dflist, ignore_index=True)

        return df_all.rename(columns={"peak_rss_MB": "memory_usage", 0: "memory_usage"})


def melt_metrics(df, metric):
    df["NLoci"] = df["NLoci"].astype(int)
    df_melt = df.melt(
        id_vars=["method", "run_id", "type", "NLoci"],
        value_vars=[metric],
        var_name="metric",
        value_name="value",
    )
    df_melt["NLoci"] = df_melt["NLoci"].astype(int)
    return df_melt.sort_values(by=["NLoci", "method", "metric"])


def plot_benchmark_results(df_long, output_dir, metric):
    output_dir.mkdir(parents=True, exist_ok=True)
    df_long["NLoci"] = df_long["NLoci"].astype(int)
    df_long = df_long.sort_values(by=["NLoci", "method", "metric"])
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
        data=df_long[df_long["method"].isin(["vcfR", "VCFReader"])],
        x="NLoci",
        y="value",
        hue="method",
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
            output_dir / f"{metric}_summary.{fmt}", dpi=300, bbox_inches="tight"
        )


if __name__ == "__main__":
    main()
