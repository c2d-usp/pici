
from experiments.runner import run_experiments


if __name__ == "__main__":
    M = 5          # number of experiments
    K = 3          # runs per experiment
    timeout = 2.0  # seconds per run

    # Choose which functions to run
    selected_funcs = ["func_a", "func_c"]  # e.g. only func_a and func_c

    run_experiments(selected_funcs, M, K, timeout)
