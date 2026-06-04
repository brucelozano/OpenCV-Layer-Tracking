import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path

from layer_velocity_rollout_config import (
    EXPERIMENT_LABEL,
    OUTPUT_ROOT_DIR,
    PROJECT_ROOT,
    TARGET_DATASETS,
)


def load_input_csv_from_params(params_file: Path) -> Path:
    spec = importlib.util.spec_from_file_location("rollout_params_module", str(params_file))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import params file: {params_file}")

    params_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(params_module)
    input_csv = getattr(params_module, "FILE_PATH", "")
    if not input_csv:
        raise RuntimeError(f"Params file missing FILE_PATH: {params_file}")
    return Path(str(input_csv))


def main() -> int:
    output_root = (OUTPUT_ROOT_DIR / EXPERIMENT_LABEL).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / ".mplconfig").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / ".cache").mkdir(parents=True, exist_ok=True)

    success = []
    failed = []

    batch_start = time.perf_counter()
    print(f"Running layer-velocity rollout for {len(TARGET_DATASETS)} datasets")
    print(f"Output root: {output_root}")

    for idx, dataset_spec in enumerate(TARGET_DATASETS, start=1):
        dataset_name = str(dataset_spec["dataset_name"])
        cruise_name = str(dataset_spec["cruise_name"])
        params_file = (PROJECT_ROOT / str(dataset_spec["params_file"])).resolve()
        reviewed_dir = (PROJECT_ROOT / str(dataset_spec["reviewed_contours_dir"])).resolve()

        print(f"\n[{idx}/{len(TARGET_DATASETS)}] {dataset_name}")
        print(f"  Cruise: {cruise_name}")
        print(f"  Params: {params_file}")

        if not params_file.exists():
            failed.append((dataset_name, "params file not found"))
            print("  Result: FAILED (params file not found)")
            continue

        try:
            input_csv = load_input_csv_from_params(params_file)
        except Exception as exc:
            failed.append((dataset_name, f"failed to load params FILE_PATH: {exc}"))
            print(f"  Result: FAILED ({exc})")
            continue

        if not input_csv.exists():
            failed.append((dataset_name, f"input CSV not found: {input_csv}"))
            print(f"  Result: FAILED (input CSV missing: {input_csv})")
            continue

        run_output_dir = output_root / cruise_name / dataset_name
        run_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Input CSV: {input_csv}")
        print(f"  Output: {run_output_dir}")

        if not reviewed_dir.exists():
            print(f"  Warning: reviewed contours directory not found: {reviewed_dir}")

        env = os.environ.copy()
        env["MPLCONFIGDIR"] = str((PROJECT_ROOT / ".mplconfig").resolve())
        env["XDG_CACHE_HOME"] = str((PROJECT_ROOT / ".cache").resolve())
        env["ECHOGRAM_PARAMS_FILE"] = str(params_file)
        env["ECHOGRAM_INPUT_FILE"] = str(input_csv)
        env["ECHOGRAM_DATASET_NAME"] = dataset_name
        env["ECHOGRAM_CRUISE_NAME"] = cruise_name
        env["ECHOGRAM_FIGURES_DIR"] = str(run_output_dir)
        env["ECHOGRAM_ENABLE_LAYER_REVIEW"] = "0"
        env["ECHOGRAM_LOAD_REVIEWED_CONTOURS"] = "1"
        env["ECHOGRAM_SKIP_LAYER_REVIEW_IF_LOADED"] = "1"
        env["ECHOGRAM_SAVE_REVIEWED_CONTOURS"] = "0"
        env["ECHOGRAM_REVIEWED_CONTOURS_DIR"] = str(reviewed_dir)

        cmd = [sys.executable, str(PROJECT_ROOT / "echogram_processing.py")]
        run_start = time.perf_counter()
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env)
        run_elapsed = time.perf_counter() - run_start

        if result.returncode == 0:
            success.append(dataset_name)
            print(f"  Result: SUCCESS ({run_elapsed:.1f}s)")
        else:
            failed.append((dataset_name, f"exit code {result.returncode}"))
            print(f"  Result: FAILED (exit code {result.returncode}, {run_elapsed:.1f}s)")

    elapsed = time.perf_counter() - batch_start
    print("\nLayer-velocity rollout complete")
    print(f"  Success: {len(success)}")
    print(f"  Failed: {len(failed)}")
    print(f"  Elapsed: {elapsed:.1f}s")

    if failed:
        print("\nFailed datasets:")
        for dataset_name, reason in failed:
            print(f"  - {dataset_name}: {reason}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
