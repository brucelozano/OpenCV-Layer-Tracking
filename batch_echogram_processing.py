import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path

from batch_master_config import (
    ALLOW_FALLBACK_TO_MASTER_CONFIG,
    CUSTOM_PARAMS_ROOT,
    DATASET_PARAMS_PREFIX,
    EXPERIMENT_LABEL,
    GROUP_OUTPUTS_BY_CRUISE,
    INPUT_CSV_DIR,
    MASTER_CONFIG_MODULE,
    ONLY_FILENAMES_CONTAINING,
    OUTPUT_ROOT_DIR,
    SKIP_FILENAMES_CONTAINING,
    USE_MASTER_CONFIG_PARAMS,
    VERBOSE_BATCH_DEBUG,
)


def strip_sv_csv_suffix(filename: str) -> str:
    if filename.endswith(".sv.csv"):
        return filename[:-7]
    return Path(filename).stem


def extract_dataset_id(filename: str) -> str:
    """
    For names like B082N_CTD254_18kHz.sv.csv -> B082N_CTD254
    Falls back to the first token if no underscore exists.
    """
    base = strip_sv_csv_suffix(filename)
    parts = base.split("_")
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return parts[0]


def module_exists(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def filename_passes_filters(filename: str) -> bool:
    if ONLY_FILENAMES_CONTAINING and not any(k in filename for k in ONLY_FILENAMES_CONTAINING):
        return False
    if SKIP_FILENAMES_CONTAINING and any(k in filename for k in SKIP_FILENAMES_CONTAINING):
        return False
    return True


def infer_cruise_name(csv_path: Path, input_root: Path) -> str:
    """
    Infer cruise folder from the path relative to the batch input root.

    For EV DVM Echograms/DP02/foo.sv.csv, this returns DP02.
    Falls back to the first filename token when there is no parent cruise folder.
    """
    try:
        relative_path = csv_path.relative_to(input_root)
        if len(relative_path.parts) > 1:
            return relative_path.parts[0]
    except ValueError:
        pass

    first_token = strip_sv_csv_suffix(csv_path.name).split("_")[0]
    return first_token


def resolve_params_source(csv_path: Path, input_root: Path) -> tuple[str, str, str]:
    """
    Returns (params_source, source_type, resolution_mode).

    source_type is one of: module, file
    resolution_mode is one of: master, custom_file, custom_module, fallback_master, missing_custom
    """
    if USE_MASTER_CONFIG_PARAMS:
        return MASTER_CONFIG_MODULE, "module", "master"

    dataset_name = strip_sv_csv_suffix(csv_path.name)
    cruise_name = infer_cruise_name(csv_path, input_root)
    custom_params_file = Path(CUSTOM_PARAMS_ROOT)
    if not custom_params_file.is_absolute():
        custom_params_file = Path(__file__).resolve().parent / custom_params_file
    custom_params_file = custom_params_file / cruise_name / f"{DATASET_PARAMS_PREFIX}{dataset_name}.py"
    if custom_params_file.exists():
        return str(custom_params_file), "file", "custom_file"

    dataset_id = extract_dataset_id(csv_path.name)
    custom_module = f"{DATASET_PARAMS_PREFIX}{dataset_id}"
    if module_exists(custom_module):
        return custom_module, "module", "custom_module"

    if ALLOW_FALLBACK_TO_MASTER_CONFIG:
        return MASTER_CONFIG_MODULE, "module", "fallback_master"

    return "", "module", "missing_custom"


def main() -> int:
    project_root = Path(__file__).resolve().parent
    input_dir = INPUT_CSV_DIR
    experiment_folder_name = EXPERIMENT_LABEL.strip() if EXPERIMENT_LABEL else "default_experiment"
    output_root = (project_root / OUTPUT_ROOT_DIR / experiment_folder_name).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print(f"Input directory does not exist: {input_dir}")
        return 1

    sv_csv_files = sorted(p for p in input_dir.rglob("*.sv.csv") if filename_passes_filters(p.name))
    if not sv_csv_files:
        print(f"No .sv.csv files found in: {input_dir}")
        return 1

    batch_start = time.perf_counter()
    print(f"Found {len(sv_csv_files)} files to process")
    print(f"Experiment label: {experiment_folder_name}")
    print(f"Parameter source mode: {'master config' if USE_MASTER_CONFIG_PARAMS else 'custom per dataset'}")
    print(f"Batch output root: {output_root}\n")

    success = []
    failed = []
    skipped = []

    for i, csv_file in enumerate(sv_csv_files, start=1):
        dataset_name = strip_sv_csv_suffix(csv_file.name)
        cruise_name = infer_cruise_name(csv_file, input_dir)
        params_source, params_source_type, mode = resolve_params_source(csv_file, input_dir)

        if mode == "missing_custom":
            reason = "missing custom params module and fallback disabled"
            print(f"[{i}/{len(sv_csv_files)}] SKIP {csv_file.name}: {reason}")
            skipped.append((csv_file.name, reason))
            continue

        if params_source_type == "module" and not module_exists(params_source):
            reason = f"params module not found: {params_source}"
            print(f"[{i}/{len(sv_csv_files)}] SKIP {csv_file.name}: {reason}")
            skipped.append((csv_file.name, reason))
            continue

        if GROUP_OUTPUTS_BY_CRUISE and cruise_name:
            run_output_dir = output_root / cruise_name / dataset_name
        else:
            run_output_dir = output_root / dataset_name
        run_output_dir.mkdir(parents=True, exist_ok=True)

        print(f"[{i}/{len(sv_csv_files)}] Processing {csv_file.name}")
        print(f"  Cruise: {cruise_name}")
        print(f"  Params source: {params_source} ({mode})")
        print(f"  Output folder: {run_output_dir}")

        env = os.environ.copy()
        if params_source_type == "file":
            env["ECHOGRAM_PARAMS_FILE"] = params_source
            env.pop("ECHOGRAM_PARAMS_MODULE", None)
        else:
            env["ECHOGRAM_PARAMS_MODULE"] = params_source
            env.pop("ECHOGRAM_PARAMS_FILE", None)
        env["ECHOGRAM_INPUT_FILE"] = str(csv_file)
        env["ECHOGRAM_DATASET_NAME"] = dataset_name
        env["ECHOGRAM_CRUISE_NAME"] = cruise_name
        env["ECHOGRAM_GROUP_OUTPUTS_BY_CRUISE"] = "1" if GROUP_OUTPUTS_BY_CRUISE else "0"
        env["ECHOGRAM_FIGURES_DIR"] = str(output_root)

        cmd = [sys.executable, str(project_root / "echogram_processing.py")]
        if VERBOSE_BATCH_DEBUG:
            print(f"  Dataset name: {dataset_name}")
            print(f"  Cruise name: {cruise_name}")
            print(f"  Input CSV: {csv_file}")
            print(f"  Command: {' '.join(cmd)}")
            if params_source_type == "file":
                print(f"  Env[ECHOGRAM_PARAMS_FILE]={env['ECHOGRAM_PARAMS_FILE']}")
            else:
                print(f"  Env[ECHOGRAM_PARAMS_MODULE]={env['ECHOGRAM_PARAMS_MODULE']}")
            print(f"  Env[ECHOGRAM_GROUP_OUTPUTS_BY_CRUISE]={env['ECHOGRAM_GROUP_OUTPUTS_BY_CRUISE']}")
            print(f"  Env[ECHOGRAM_FIGURES_DIR]={env['ECHOGRAM_FIGURES_DIR']}")

        run_start = time.perf_counter()
        result = subprocess.run(cmd, cwd=str(project_root), env=env)
        run_elapsed = time.perf_counter() - run_start

        if result.returncode == 0:
            success.append(csv_file.name)
            print(f"  Result: SUCCESS ({run_elapsed:.1f}s)\n")
        else:
            failed.append((csv_file.name, result.returncode))
            print(f"  Result: FAILED (exit code {result.returncode}, {run_elapsed:.1f}s)")
            if VERBOSE_BATCH_DEBUG:
                print("  Debug hint: Scroll up to the most recent traceback above this result.")
            print()

    total_elapsed = time.perf_counter() - batch_start
    print("\nBatch complete")
    print(f"  Success: {len(success)}")
    print(f"  Failed: {len(failed)}")
    print(f"  Skipped: {len(skipped)}")
    print(f"  Elapsed: {total_elapsed:.1f}s")

    if failed:
        print("\nFailed files:")
        for filename, code in failed:
            print(f"  - {filename}: exit code {code}")

    if skipped:
        print("\nSkipped files:")
        for filename, reason in skipped:
            print(f"  - {filename}: {reason}")

    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
