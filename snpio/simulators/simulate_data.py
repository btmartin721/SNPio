import random

import msprime
import pysam
import numpy as np
import pandas as pd
import demes

from snpio.simulators.simulate import SNPulator, SNPulatoRate, SNPulatorConfig


def demographic_model(
    n_pops=3,
    num_loci=100,
    num_samples=100,
    random_seed=None,
    na=1500,
    min_log_mig=-1.5,
    max_log_mig=1.5,
    growth_rate=0.01,  # 1% bottleneck per generation
    growth_rate_anc=-0.01,  # 1% growth per generation
    time=45,  # 45 generations
    recovery_ne=1500,
    bottleneck_ne=50,
    debug=False,
):
    # # Parameters
    recovery_ne = 1500  # Effective size after recovery

    # Convert log migration rate to linear scale
    def log_to_linear(log_rate):
        return np.exp(log_rate) / (2 * na)

    min_lin_mig = log_to_linear(min_log_mig)
    max_lin_mig = log_to_linear(max_log_mig)

    # Initialize demography
    demography = msprime.Demography()

    for pop in range(1, n_pops + 1):
        # Add a couple of populations
        demography.add_population(name=f"Pop{pop}", initial_size=na)

    # Assign samples to these populations
    samples = [
        msprime.SampleSet(population=pop, num_samples=num_samples)
        for pop in range(n_pops)
    ]

    for pop in range(1, n_pops + 1):
        if pop < n_pops:
            demography.set_symmetric_migration_rate(
                populations=[f"Pop{pop}", f"Pop{pop + 1}"],
                rate=np.random.uniform(low=min_lin_mig, high=max_lin_mig, size=1)
                / (2 * na),  # scale to per-generation.
            )

        demography.add_population_parameters_change(
            time=0,
            initial_size=recovery_ne,
            growth_rate=growth_rate,
            population=f"Pop{pop}",
        )

        demography.add_population_parameters_change(
            time=time,
            initial_size=bottleneck_ne,
            growth_rate=growth_rate_anc,
            population=f"Pop{pop}",
        )

        demography.add_population_parameters_change(
            time=90,
            initial_size=recovery_ne,
            growth_rate=0.0,
            population=f"Pop{pop}",
        )

    demography.sort_events()

    if debug:
        # Debug the demography
        print(demography.debug())
        return None

    return demography


def introduce_missing_data(input_vcf, output_vcf, missing_ratio=0.2):
    # Open the VCF file
    vcf_in = pysam.VariantFile(input_vcf, "r")
    vcf_out = pysam.VariantFile(output_vcf, "w", header=vcf_in.header)

    # Calculate total number of genotype entries and determine how many to replace
    total_entries = sum(len(record.samples) for record in vcf_in)
    num_missing = int(total_entries * missing_ratio)

    # Generate random positions to replace with missing data
    missing_positions = set(random.sample(range(total_entries), num_missing))

    # Reset file pointer to the beginning of the file
    vcf_in.seek(0)

    # Counter for current position
    current_pos = 0

    for record in vcf_in:
        for sample in record.samples.values():
            # Replace with missing data if this position is selected
            if current_pos in missing_positions:
                sample["GT"] = (None, None)
            current_pos += 1

        # Write the modified record to the new VCF
        vcf_out.write(record)

    vcf_in.close()
    vcf_out.close()


# Run the simulation function
demography = demographic_model(
    n_pops=4,
    num_loci=436,
    num_samples=200,
    random_seed=42,
    na=1500,
    min_log_mig=-1.5,
    max_log_mig=1.5,
    debug=False,
    growth_rate=0.075,
    growth_rate_anc=-0.075,
)

g = demography.to_demes()
demes.dump(g, "/Users/btm002/Documents/wtd/GeoGenIE/data/demography.yaml")


# if tree_sequence is not None:
#     # Example usage
#     introduce_missing_data(
#         "data/simulated_data.vcf",
#         "data/modified_simulated_data.vcf",
#         missing_ratio=0.2,
#     )
