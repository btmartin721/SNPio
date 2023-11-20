import matplotlib.pylab as plt
import numpy as np

import demes
import demesdraw


def generate_random_parameters_for_nine_pops():
    params = {}

    params["Ne"] = np.random.uniform(1e3, 1e4)

    # Randomly sample effective population sizes
    for i in range(1, 10):
        params[f"nu{i}"] = np.random.uniform(1e3, 1e4)

    # Randomly sample migration rates for each pair of populations
    for i in range(1, 10):
        for j in range(i + 1, 10):
            params[f"m{i}{j}"] = np.random.uniform(0, 0.1)

    return params


def generate_random_parameters():
    """Generate random values for a two-population demes model with two symmetric migration events and size changes.

    Returns:
        dict: Dictionary containing random parameter values. Keys are the parameter names and values are the randomly generated floats.

    """
    # Randomly sample effective population sizes
    Ne = np.random.uniform(1000, 100000)
    nu1a = np.random.uniform(1000, 100000)
    nu2a = np.random.uniform(1000, 100000)
    nu1b = np.random.uniform(1000, 100000)
    nu2b = np.random.uniform(1000, 100000)

    # Randomly sample migration rates
    ma = np.random.uniform(0, 0.1)
    mb = np.random.uniform(0, 0.1)

    # Randomly sample divergence times
    T1 = np.random.uniform(50000, 100000)
    T2 = np.random.uniform(1000, 40000)

    # Create a dictionary to hold these random parameters
    params = {
        "Ne": Ne,
        "nu1a": nu1a,
        "nu2a": nu2a,
        "nu1b": nu1b,
        "nu2b": nu2b,
        "ma": ma,
        "mb": mb,
        "T1": T1,
        "T2": T2,
    }

    return params


def generate_random_parameters2():
    Ne = np.random.uniform(1000)
    nu1 = np.random.uniform(1000, 1000)
    nu2 = np.random.uniform(1000, 1000)
    T1 = 1e7

    params = {
        "Ne": Ne,
        "nu1": nu1,
        "nu2": nu2,
        "T1": T1,
    }

    return params


def two_pop_sym_mig_size(params):
    """Two-population demes model with two symmetric migration events and size changes.

    Args:
        Ne (float): Ancestral population size.
        nu1a (float): Ancestral population size of population 1.
        nu2a (float): Ancestral population size of population 2.
        nu1b (float): T2 population size for population 1.
        nu2b (float) T2 population size for population 2.
        ma (float); Migration rate for first epoch (T1).
        mb (float): Migration rate for second epoch (T2).
        T1 (float): Divergence time for split between population 1 and population 2.
        T2 (float): Time of second epoch.

    Returns:
        demes graph: Demes graph object.
    """
    # 9 parameters.
    Ne, nu1a, nu2a, nu1b, nu2b, ma, mb, T1, T2 = params.values()

    b = demes.Builder(
        description="Two populations with size changes and two migration events.",
        time_units="years",
        generation_time=1,
    )

    b.add_deme(
        "ancestral", epochs=[dict(end_time=T1, start_size=Ne, end_size=Ne)]
    )

    b.add_deme(
        name="ON_T1",
        ancestors=["ancestral"],
        start_time=T1,
        epochs=[dict(end_time=T2, end_size=nu1a, start_size=nu1a)],
    )
    b.add_deme(
        name="DS_T1",
        ancestors=["ancestral"],
        start_time=T1,
        epochs=[dict(end_time=T2, end_size=nu2b, start_size=nu2b)],
    )

    b.add_deme(
        name="ON_T2",
        ancestors=["ON_T1"],
        start_time=T2,
        epochs=[dict(start_size=nu1a, end_size=nu1b)],
    )

    b.add_deme(
        name="DS_T2",
        ancestors=["DS_T1"],
        start_time=T2,
        epochs=[dict(start_size=nu2a, end_size=nu2b)],
    )

    b.add_migration(
        demes=["ON_T1", "DS_T1"], start_time=T1, end_time=T2, rate=ma
    )

    b.add_migration(
        demes=["ON_T2", "DS_T2"], start_time=T2, end_time=0, rate=mb
    )

    graph = b.resolve()

    ax = demesdraw.tubes(
        graph,
        title="Two populations with two symmetric migration events and size changes.",
        scale_bar=True,
        labels="xticks-mid",
    )

    ax.axhline(y=T2, c="b", ls="--", lw=1, label="Bottleneck Event")
    return graph


def simple_example_2pop(params):
    """Two-population demes model with no migration.

    Args:
        Ne (float): Ancestral population size.
        nu1 (float): Size of population 1.
        nu2 (float): Size of population 2.
        T1 (float): Divergence time for split between population 1 and population 2.

    Returns:
        demes graph: Demes graph object.
    """
    # 9 parameters.
    Ne, nu1, nu2, T1 = params.values()

    b = demes.Builder(
        description="Two populations with no migration.",
        time_units="years",
        generation_time=1,
    )

    b.add_deme(
        "ancestral", epochs=[dict(end_time=T1, start_size=Ne, end_size=Ne)]
    )

    b.add_deme(
        "pop1",
        ancestors=["ancestral"],
        epochs=[dict(start_size=nu1, end_size=nu1)],
    )

    b.add_deme(
        "pop2",
        ancestors=["ancestral"],
        epochs=[dict(start_size=nu2, end_size=nu2)],
    )

    graph = b.resolve()

    ax = demesdraw.tubes(
        graph,
        title="Two populations with no migration",
        scale_bar=True,
        labels="xticks-mid",
    )

    return graph


def nine_pop_sym_mig_size(params):
    """
    Nine-population demes model with symmetric migration events and size changes.

    Args:
        params (dict): Dictionary containing parameter values.

    Returns:
        demes graph: Demes graph object.
    """
    b = demes.Builder(
        description="Nine populations with size changes and migration events.",
        time_units="years",
        generation_time=1,
    )

    T = [1e6, 2e6, 3e6, 4e6, 5e6, 1e7, 2e7, 3e7, 4e7, 5e7]
    T.reverse()

    Ne = params["Ne"]

    # Add ancestral deme with end_time set to max_Ti.
    b.add_deme(
        "ancestral",
        epochs=[dict(end_time=max(T), start_size=Ne, end_size=Ne)],
    )
    # Add ancestral deme with end_time set to max_Ti.
    b.add_deme(
        "pop1",
        ancestors=["ancestral"],
        start_time=max(T),
        epochs=[
            dict(end_time=0, start_size=params["nu1"], end_size=params["nu1"])
        ],
    )
    # Add ancestral deme with end_time set to max_Ti.
    b.add_deme(
        "pop2",
        ancestors=["ancestral"],
        start_time=max(T),
        epochs=[
            dict(end_time=0, start_size=params["nu2"], end_size=params["nu2"])
        ],
    )

    # Derived populations
    for i in range(3, 10):
        b.add_deme(
            f"pop{i}",
            ancestors=[f"pop{i - 1}"],
            start_time=T[i],
            epochs=[
                dict(
                    end_time=0,
                    start_size=params[f"nu{i}"],
                    end_size=params[f"nu{i}"],
                )
            ],
        )

    # Add migration events between each pair of the 9 populations
    # Doing secondary contact here with start_time=T[j] / 2.
    for i in range(1, 10):
        for j in range(i + 1, 10):
            ma = params[f"m{i}{j}"]
            b.add_migration(
                demes=[f"pop{i}", f"pop{j}"],
                start_time=1e4,
                end_time=0,
                rate=ma,
            )

    graph = b.resolve()

    ax = demesdraw.tubes(
        graph,
        title="Nine populations with symmetric migration events and size changes.",
        scale_bar=True,
        labels="xticks-mid",
        log_time=True,
        num_lines_per_migration=2,
    )
    plt.show()

    return graph


# Example params dictionary (for demonstration; you would typically generate this randomly)
params = {
    "Ne": 50000,
    **{f"nu{i}a": 10000 for i in range(1, 10)},
    **{f"nu{i}b": 5000 for i in range(1, 10)},
    **{f"m{i}{j}a": 0.05 for i in range(1, 10) for j in range(i + 1, 10)},
    **{f"m{i}{j}b": 0.02 for i in range(1, 10) for j in range(i + 1, 10)},
    **{f"T{i}": 90000 - i * 2000 for i in range(1, 10)},
}


g = two_pop_sym_mig_size(generate_random_parameters())
demes.dump(g, "./example_demes.yaml", simplified=True)

g2 = simple_example_2pop(generate_random_parameters2())
demes.dump(g2, "./example_demes2.yaml")

# Generate the demes graph
g3 = nine_pop_sym_mig_size(generate_random_parameters_for_nine_pops())
demes.dump(g3, "SNPio/example_data/demography/example_demes3.yaml")
