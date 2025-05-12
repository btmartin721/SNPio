import gc
import json
import time
from collections import defaultdict
from pathlib import Path

from custom_inherit import store
from memory_profiler import memory_usage
from tqdm import tqdm

from snpio import NRemover2, VCFReader
from snpio.utils.benchmarking import Benchmark


def benchmark_io(n_loci):
    vcf_reader = VCFReader(
        filename=f"dryad_files/benchmarking_inputs/bmtest_{n_loci}.vcf.gz",
        popmapfile="dryad_files/benchmarking_full/nloci_test.popmap.txt",
        force_popmap=True,
        plot_format="png",
        prefix=f"test_io_vcf_{n_loci}_hf",
        chunk_size=1000 if n_loci == 1000 else 5000,
    )
    return


def run_nremover_benchmarks(repeats, loci_count, output_dir):

    input_file = Path(f"dryad_files/benchmarking_inputs/bmtest_{loci_count}.vcf.gz")

    gd = VCFReader(
        filename=input_file,
        popmapfile="dryad_files/benchmarking_full/nloci_test.popmap.txt",
        force_popmap=True,
        plot_format="png",
        prefix=f"test_nremover_vcf_{loci_count}",
        chunk_size=50000,
    )

    benchmark_steps = [
        ("Filter Missing", NRemover2(gd.copy()).filter_missing, 0.5),
        ("Filter MAF", NRemover2(gd.copy()).filter_maf, 0.01),
        ("Filter MAC", NRemover2(gd.copy()).filter_mac, 2),
        ("Filter Monomorphic", NRemover2(gd.copy()).filter_monomorphic, False),
        ("Filter Biallelic", NRemover2(gd.copy()).filter_biallelic, False),
        ("Filter Singletons", NRemover2(gd.copy()).filter_singletons, False),
        ("Thin Loci", NRemover2(gd.copy()).thin_loci, 100),
        ("Filter Missing (Sample)", NRemover2(gd.copy()).filter_missing_sample, 0.8),
        ("Filter Missing (Pop)", NRemover2(gd.copy()).filter_missing_pop, 0.5),
        ("Random Subset Loci", NRemover2(gd.copy()).random_subset_loci, 500),
    ]

    for label, method, threshold in benchmark_steps:
        metrics = Benchmark.run_repeated_subprocess(
            name=label,
            func=nremover_pipeline,
            repeats=repeats,
            func2=method,
            threshold=threshold,
        )
        nrm = NRemover2(gd.copy())
        nrm.resource_data = {label: metrics}

        output_dir_child = output_dir / "results_refactor"
        output_dir_child.mkdir(parents=True, exist_ok=True)

        Benchmark.save_performance(
            resource_data=nrm.resource_data,
            objs=[nrm],
            save_dir=output_dir_child,
            outfile_prefix=f"{label}_{loci_count}",
        )
    print(f"NRemover {label} benchmarking results saved to: {output_dir_child}")


def io_pipeline(IOReader, **kwargs):
    reader = IOReader(**kwargs)


def ensure_output_directory(output_dir):
    if not isinstance(output_dir, Path):
        outdir = Path(output_dir)
    else:
        outdir = output_dir
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def nremover_pipeline(func, threshold, gd):
    """Runs a filtering function on a copy of GenotypeData."""
    gd_copy = gd.copy()
    nrm = NRemover2(gd_copy)
    filt = func(nrm, threshold).resolve(benchmark_mode=True)


def plot_benchmarks(outdir, *args, prefix=None, method_type="io"):
    all_resources = {}
    for obj in args:
        for name, entries in getattr(obj, "resource_data", {}).items():
            all_resources.setdefault(name, []).extend(entries)

    outdir = outdir / "plots_refactor"
    outdir.mkdir(parents=True, exist_ok=True)

    Benchmark.plot_performance(
        resource_data=all_resources,
        save_dir=outdir,
        outfile_prefix=prefix,
        metrics_to_plot=["execution_time", "memory_footprint"],
        show=False,
        output_formats=["png", "pdf"],
        method_type=method_type,
    )
    return outdir


def main():
    output_dir = ensure_output_directory(
        "dryad_files/benchmarking_results_refactor/results_refactor"
    )

    n_replicates = 100
    warmup = 2

    loci_sizes = [1000, 5000, 10000, 50000, 100000]
    loci_sizes = [10000]

    # Mapping of filter name to (function accessor, threshold)
    filter_map = {
        "filter_missing": (lambda nrm, x: nrm.filter_missing(x), 0.5),
        "filter_missing_sample": (lambda nrm, x: nrm.filter_missing_sample(x), 0.5),
        "filter_missing_pop": (lambda nrm, x: nrm.filter_missing_pop(x), 0.5),
        "filter_maf": (lambda nrm, x: nrm.filter_maf(x), 0.01),
        "filter_mac": (lambda nrm, x: nrm.filter_mac(x), 2),
        "filter_biallelic": (lambda nrm, x: nrm.filter_biallelic(x), False),
        "filter_monomorphic": (lambda nrm, x: nrm.filter_monomorphic(x), False),
        "filter_singletons": (lambda nrm, x: nrm.filter_singletons(x), False),
        "thin_loci": (lambda nrm, x: nrm.thin_loci(x), 100),
        "random_subset_loci": (lambda nrm, x: nrm.random_subset_loci(x), 100),
    }

    for n_loci in loci_sizes:
        print(f"\nBenchmarking filters on {n_loci} loci...")
        mem_dict = defaultdict(dict)

        gd = VCFReader(
            filename=f"dryad_files/benchmarking_inputs/bmtest_{n_loci}.vcf.gz",
            popmapfile="dryad_files/benchmarking_full/nloci_test.popmap.txt",
            force_popmap=True,
            plot_format="png",
            prefix=f"test_io_vcf_{n_loci}_memtest",
            chunk_size=1000 if n_loci == 1000 else 5000,
        )

        import sys

        sys.exit()

        for replicate in tqdm(
            range(n_replicates + warmup), desc=f"{n_loci} loci replicates"
        ):
            for filt_name, (filter_func, threshold) in filter_map.items():
                try:
                    mem = memory_usage(
                        proc=(nremover_pipeline, (filter_func, threshold, gd)),
                        max_usage=True,
                    )

                    if replicate >= warmup:
                        mem_dict[filt_name][replicate - warmup] = mem

                except Exception as e:
                    print(f"Error in {filt_name} (replicate {replicate}): {e}")
                    continue

                time.sleep(1.0)
                gc.collect()

        out_file = output_dir / f"nremover_memory_usage_{n_loci}_loci.json"
        with out_file.open("w") as f:
            json.dump(mem_dict, f, indent=4)

        print(f"Saved memory usage results for {n_loci} loci to: {out_file}")
        del mem_dict, gd
        gc.collect()


if __name__ == "__main__":
    main()
