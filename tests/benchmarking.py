import sys
import time
import psutil
import os
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append("C:/Users/evobi/Desktop/SNPio")
from snpio.read_input.genotype_data import GenotypeData


# Function to measure CPU load
def measure_cpu_load():
    cpu_load = psutil.cpu_percent()
    return cpu_load


# Function to measure memory footprint
def measure_memory_footprint():
    process = psutil.Process(os.getpid())
    memory_footprint = process.memory_info().rss
    memory_footprint_mb = memory_footprint / (
        1024 * 1024
    )  # Convert bytes to megabytes
    return memory_footprint_mb


# Function to measure execution time
def measure_execution_time(start_time):
    end_time = time.time()
    execution_time = end_time - start_time
    return execution_time


# Function to plot resource usage and execution time
def plot_performance(resource_data, execution_time_data, title, fontsize=14):
    methods = list(resource_data.keys())

    cpu_loads = [data["cpu_load"] for data in resource_data.values()]
    memory_footprints = [
        data["memory_footprint"] for data in resource_data.values()
    ]
    execution_times = [
        data["execution_time"] for data in execution_time_data.values()
    ]

    # Plot CPU Load
    fig, axs = plt.subplots(1, 3, figsize=(16, 9))
    plt.sca(axs[0])

    sns.barplot(
        x=methods,
        y=cpu_loads,
        errorbar=None,
    )
    plt.xlabel("Methods", fontsize=fontsize)
    plt.ylabel("CPU Load (%)", fontsize=fontsize)
    plt.title(f"CPU Load for {title}", fontsize=fontsize)
    plt.xticks(rotation=90, fontsize=fontsize)
    plt.ylim(bottom=0)
    plt.tight_layout()

    plt.sca(axs[1])

    # Plot Memory Footprint
    sns.lineplot(
        x=methods,
        y=memory_footprints,
        errorbar=None,
    )
    plt.xlabel("Method Execution/ Property Access", fontsize=fontsize)
    plt.ylabel("Memory Footprint (MB)", fontsize=fontsize)
    plt.title(f"Memory Footprint for {title}", fontsize=fontsize)
    plt.xticks(rotation=90, fontsize=fontsize)
    plt.tight_layout()

    plt.sca(axs[2])

    # Plot Execution Time
    sns.barplot(
        x=methods,
        y=execution_times,
        errorbar=None,
    )
    plt.xlabel("Methods", fontsize=fontsize)
    plt.ylabel("Execution Time (seconds)", fontsize=fontsize)
    plt.title(f"Execution Time for {title}", fontsize=fontsize)
    plt.xticks(rotation=90, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.tight_layout()

    fig.savefig(f"tests/benchmarking_plot.png", facecolor="white")


resource_data = {}
execution_time_data = {}

# Measure performance for each property and method
start_time = time.time()

# Instantiate GenotypeData object
genotype_data = GenotypeData(
    filename="example_data/vcf_files/phylogen_subset14K.vcf.gz",
    force_popmap=True,
    filetype="auto",
    popmapfile="example_data/popmaps/phylogen_nomx.popmap",
    guidetree="example_data/trees/test.tre",
    siterates_iqtree="example_data/trees/test14K.rate",
    qmatrix_iqtree="example_data/trees/test.iqtree",
)

cpu_load = measure_cpu_load()
memory_footprint = measure_memory_footprint()
execution_time = measure_execution_time(start_time)

resource_data["Initialization"] = {
    "cpu_load": cpu_load,
    "memory_footprint": memory_footprint,
}
execution_time_data["Initialization"] = {"execution_time": execution_time}

start_time = time.time()

# Access basic properties
num_snps = genotype_data.num_snps
num_inds = genotype_data.num_inds
populations = genotype_data.populations
popmap = genotype_data.popmap
samples = genotype_data.samples

cpu_load = measure_cpu_load()
memory_footprint = measure_memory_footprint()
execution_time = measure_execution_time(start_time)

resource_data["Access Basic Properties"] = {
    "cpu_load": cpu_load,
    "memory_footprint": memory_footprint,
}
execution_time_data["Access Basic Properties"] = {
    "execution_time": execution_time
}

start_time = time.time()

# Access 012 genotypes
genotypes_012 = genotype_data.genotypes_012(fmt="list")

cpu_load = measure_cpu_load()
memory_footprint = measure_memory_footprint()
execution_time = measure_execution_time(start_time)

resource_data["Access 012 Genotypes"] = {
    "cpu_load": cpu_load,
    "memory_footprint": memory_footprint,
}
execution_time_data["Access 012 Genotypes"] = {
    "execution_time": execution_time
}

start_time = time.time()

# Access other transformed data
genotypes_onehot = genotype_data.genotypes_onehot
genotypes_int = genotype_data.genotypes_int
alignment = genotype_data.alignment

cpu_load = measure_cpu_load()
memory_footprint = measure_memory_footprint()
execution_time = measure_execution_time(start_time)

resource_data["Access Other Transformations"] = {
    "cpu_load": cpu_load,
    "memory_footprint": memory_footprint,
}
execution_time_data["Access Other Transformations"] = {
    "execution_time": execution_time
}

start_time = time.time()

# Access VCF file attributes
vcf_attributes = genotype_data.vcf_attributes

cpu_load = measure_cpu_load()
memory_footprint = measure_memory_footprint()
execution_time = measure_execution_time(start_time)

resource_data["Access VCF Attributes"] = {
    "cpu_load": cpu_load,
    "memory_footprint": memory_footprint,
}
execution_time_data["Access VCF Attributes"] = {
    "execution_time": execution_time
}

start_time = time.time()

# Access additional properties
q_matrix = genotype_data.q
site_rates = genotype_data.site_rates
newick_tree = genotype_data.tree

cpu_load = measure_cpu_load()
memory_footprint = measure_memory_footprint()
execution_time = measure_execution_time(start_time)

resource_data["Access Additional Properties"] = {
    "cpu_load": cpu_load,
    "memory_footprint": memory_footprint,
}
execution_time_data["Access Additional Properties"] = {
    "execution_time": execution_time
}

plot_performance(resource_data, execution_time_data, "Benchmarking Results")
