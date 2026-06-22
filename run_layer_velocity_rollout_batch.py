import importlib.util
import json
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


def load_reviewed_contours_status(status_path: Path):
    if not status_path.exists():
        return None
    try:
        with status_path.open("r", encoding="utf-8") as status_file:
            payload = json.load(status_file)
    except Exception as exc:
        print(f"  Warning: failed to read reviewed contour status file: {status_path} ({exc})")
        return None

    if not isinstance(payload, dict):
        print(f"  Warning: reviewed contour status payload is not a JSON object: {status_path}")
        return None
    return payload


def main() -> int:
    output_root = (OUTPUT_ROOT_DIR / EXPERIMENT_LABEL).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / ".mplconfig").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / ".cache").mkdir(parents=True, exist_ok=True)

    success = []
    failed = []
    reviewed_loaded = []
    reviewed_refused = []
    reviewed_unknown = []

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
            reviewed_unknown.append((dataset_name, "not run (params file not found)"))
            print("  Result: FAILED (params file not found)")
            continue

        try:
            input_csv = load_input_csv_from_params(params_file)
        except Exception as exc:
            failed.append((dataset_name, f"failed to load params FILE_PATH: {exc}"))
            reviewed_unknown.append((dataset_name, "not run (failed to load params FILE_PATH)"))
            print(f"  Result: FAILED ({exc})")
            continue

        if not input_csv.exists():
            failed.append((dataset_name, f"input CSV not found: {input_csv}"))
            reviewed_unknown.append((dataset_name, "not run (input CSV missing)"))
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
        env["ECHOGRAM_GROUP_OUTPUTS_BY_CRUISE"] = "1"
        env["ECHOGRAM_FIGURES_DIR"] = str(output_root)
        env["ECHOGRAM_ENABLE_LAYER_REVIEW"] = "0"
        env["ECHOGRAM_LOAD_REVIEWED_CONTOURS"] = "1"
        env["ECHOGRAM_SKIP_LAYER_REVIEW_IF_LOADED"] = "1"
        env["ECHOGRAM_SAVE_REVIEWED_CONTOURS"] = "0"
        env["ECHOGRAM_REVIEWED_CONTOURS_DIR"] = str(reviewed_dir)

        cmd = [sys.executable, str(PROJECT_ROOT / "echogram_processing.py")]
        run_start = time.perf_counter()
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env)
        run_elapsed = time.perf_counter() - run_start

        status_path = run_output_dir / "reviewed_contours_status.json"
        reviewed_status = load_reviewed_contours_status(status_path)
        if reviewed_status is None:
            reviewed_unknown.append((dataset_name, f"status file missing: {status_path}"))
            print(f"  Reviewed contours: UNKNOWN (status file missing: {status_path.name})")
        else:
            requested_load = bool(reviewed_status.get("requested_load", False))
            loaded_from_artifact = bool(reviewed_status.get("loaded_from_artifact", False))
            load_status = str(reviewed_status.get("load_status", "unknown"))
            load_reason = str(reviewed_status.get("load_reason") or "").strip()
            artifact_path = str(reviewed_status.get("artifact_path") or "")
            metadata_mismatch_reasons = reviewed_status.get("metadata_mismatch_reasons") or []

            if loaded_from_artifact:
                reviewed_loaded.append(dataset_name)
                print(f"  Reviewed contours: LOADED ({artifact_path})")
            elif requested_load:
                reason_parts = []
                if load_reason:
                    reason_parts.append(load_reason)
                if metadata_mismatch_reasons:
                    mismatch_text = ", ".join(str(reason) for reason in metadata_mismatch_reasons)
                    reason_parts.append(f"metadata={mismatch_text}")
                refusal_reason = "; ".join(reason_parts) if reason_parts else load_status
                reviewed_refused.append((dataset_name, refusal_reason, artifact_path))
                print(f"  Reviewed contours: REFUSED ({refusal_reason})")
                if artifact_path:
                    print(f"    Artifact path: {artifact_path}")
            else:
                reviewed_unknown.append((dataset_name, f"load not requested ({load_status})"))
                print(f"  Reviewed contours: NOT REQUESTED ({load_status})")

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

    print("\nReviewed contour reuse summary")
    print(f"  Loaded from artifact: {len(reviewed_loaded)}")
    print(f"  Refused while requested: {len(reviewed_refused)}")
    if reviewed_refused:
        print("  Refused datasets:")
        for dataset_name, reason, artifact_path in reviewed_refused:
            artifact_suffix = f" [artifact: {artifact_path}]" if artifact_path else ""
            print(f"  - {dataset_name}: {reason}{artifact_suffix}")

    if reviewed_unknown:
        print("  Unknown/not-run reviewed contour status:")
        for dataset_name, reason in reviewed_unknown:
            print(f"  - {dataset_name}: {reason}")

    if failed:
        print("\nFailed datasets:")
        for dataset_name, reason in failed:
            print(f"  - {dataset_name}: {reason}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
