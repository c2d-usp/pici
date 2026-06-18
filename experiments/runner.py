import pandas as pd
import time
import traceback
from multiprocessing import Process, Manager
from pathlib import Path
from datetime import datetime
from functions import func_a, func_b, func_c

FUNCTIONS = {
    "func_a": func_a,
    "func_b": func_b,
    "func_c": func_c
}

def run_with_timeout(func, args, return_dict):
    """Run a function safely and capture runtime + errors."""
    try:
        start = time.time()
        result = func(*args)
        runtime = time.time() - start
        return_dict["result"] = result
        return_dict["runtime"] = runtime
        return_dict["status"] = "ok"
    except Exception as e:
        return_dict["result"] = None
        return_dict["runtime"] = None
        return_dict["status"] = f"error: {e}"
        return_dict["traceback"] = traceback.format_exc()


def run_experiments(selected_funcs, M, K, timeout_sec, save_dir="results"):
    Path(save_dir).mkdir(exist_ok=True)
    for func_name in selected_funcs:
        print(f"Running experiments for {func_name}...")
        f = FUNCTIONS[func_name]
        df_path = Path(save_dir) / f"{func_name}.csv"

        # Resume if possible
        if df_path.exists():
            df = pd.read_csv(df_path)
        else:
            df = pd.DataFrame(columns=[
                "experiment", "run", "timestamp", "status",
                "runtime", "min_val", "max_val", "min_iter", "max_iter"
            ])

        for m in range(1, M + 1):
            for k in range(1, K + 1):
                if ((df["experiment"] == m) & (df["run"] == k)).any():
                    continue

                print(f" -> {func_name}: Experiment {m}, Run {k}")
                manager = Manager()
                return_dict = manager.dict()
                process = Process(target=run_with_timeout, args=(f, (m, k), return_dict))

                start_time = time.time()
                process.start()
                process.join(timeout=timeout_sec)
                elapsed = time.time() - start_time

                if process.is_alive():
                    process.terminate()
                    process.join()
                    row = {
                        "experiment": m,
                        "run": k,
                        "timestamp": datetime.now(),
                        "status": f"timeout (> {timeout_sec:.1f}s)",
                        "runtime": elapsed,
                        "min_val": None,
                        "max_val": None,
                        "min_iter": None,
                        "max_iter": None,
                    }
                else:
                    if return_dict.get("status") == "ok":
                        runtime = return_dict["runtime"]
                        status = "ok" if runtime <= timeout_sec else f"exceeded ({runtime:.2f}s)"
                        row = {
                            "experiment": m,
                            "run": k,
                            "timestamp": datetime.now(),
                            "status": status,
                            "runtime": runtime,
                            "min_val": return_dict["result"]["min_val"],
                            "max_val": return_dict["result"]["max_val"],
                            "min_iter": return_dict["result"]["min_iter"],
                            "max_iter": return_dict["result"]["max_iter"],
                        }
                    else:
                        row = {
                            "experiment": m,
                            "run": k,
                            "timestamp": datetime.now(),
                            "status": return_dict.get("status"),
                            "runtime": return_dict.get("runtime"),
                            "min_val": None,
                            "max_val": None,
                            "min_iter": None,
                            "max_iter": None,
                        }

                df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
                df.to_csv(df_path, index=False)

    print("✅ All experiments completed.")
