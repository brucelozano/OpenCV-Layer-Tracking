import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def _path_from_env(var_name: str, default_path: str) -> Path:
    raw_value = os.getenv(var_name, "").strip()
    return Path(raw_value).expanduser() if raw_value else Path(default_path)


OUTPUT_ROOT_DIR = _path_from_env("ECHOGRAM_ROLLOUT_OUTPUT_ROOT_DIR", "Batch Results")
EXPERIMENT_LABEL = os.getenv("ECHOGRAM_ROLLOUT_EXPERIMENT_LABEL", "layer_velocity_rollout_v1")
EV_DVM_PARAMS_DIR = _path_from_env("ECHOGRAM_ROLLOUT_EV_PARAMS_DIR", "EV DVM Params")
DP09_PARAMS_DIR = _path_from_env("ECHOGRAM_ROLLOUT_DP09_PARAMS_DIR", "DP09_CTD_params")
REVIEWED_CONTOURS_ROOT = _path_from_env("ECHOGRAM_ROLLOUT_REVIEWED_ROOT", "Figures")


TARGET_DATASETS = [
    {
        "dataset_name": "DP06_B065D_38kHz",
        "cruise_name": "DP06",
        "params_file": str(EV_DVM_PARAMS_DIR / "DP06" / "params_DP06_B065D_38kHz.py"),
        "reviewed_contours_dir": str(
            REVIEWED_CONTOURS_ROOT / "DP06" / "DP06_B065D_38kHz" / "DP06_B065D_38kHz_reviewed"
        ),
    },
    {
        "dataset_name": "DP03_B082D_18kHz",
        "cruise_name": "DP03",
        "params_file": str(EV_DVM_PARAMS_DIR / "DP03" / "params_DP03_B082D_18kHz.py"),
        "reviewed_contours_dir": str(
            REVIEWED_CONTOURS_ROOT / "DP03" / "DP03_B082D_18kHz" / "DP03_B082D_18kHz_reviewed"
        ),
    },
    {
        "dataset_name": "DP03_B082N_18kHz",
        "cruise_name": "DP03",
        "params_file": str(EV_DVM_PARAMS_DIR / "DP03" / "params_DP03_B082N_18kHz.py"),
        "reviewed_contours_dir": str(
            REVIEWED_CONTOURS_ROOT / "DP03" / "DP03_B082N_18kHz" / "DP03_B082N_18kHz_reviewed"
        ),
    },
    {
        "dataset_name": "DP03_B252D_38kHz",
        "cruise_name": "DP03",
        "params_file": str(EV_DVM_PARAMS_DIR / "DP03" / "params_DP03_B252D_38kHz.py"),
        "reviewed_contours_dir": str(
            REVIEWED_CONTOURS_ROOT / "DP03" / "DP03_B252D_38kHz" / "DP03_B252D_38kHz_reviewed"
        ),
    },
    {
        "dataset_name": "DP02_B080D_38kHz",
        "cruise_name": "DP02",
        "params_file": str(EV_DVM_PARAMS_DIR / "DP02" / "params_DP02_B080D_38kHz.py"),
        "reviewed_contours_dir": str(
            REVIEWED_CONTOURS_ROOT / "DP02" / "DP02_B080D_38kHz" / "DP02_B080D_38kHz_reviewed"
        ),
    },
    {
        "dataset_name": "DP06_B286N_38kHz",
        "cruise_name": "DP06",
        "params_file": str(EV_DVM_PARAMS_DIR / "DP06" / "params_DP06_B286N_38kHz.py"),
        "reviewed_contours_dir": str(
            REVIEWED_CONTOURS_ROOT / "DP06" / "DP06_B286N_38kHz" / "DP06_B286N_38kHz_reviewed"
        ),
    },
    {
        "dataset_name": "DP06_B251N_38kHz",
        "cruise_name": "DP06",
        "params_file": str(EV_DVM_PARAMS_DIR / "DP06" / "params_DP06_B251N_38kHz.py"),
        "reviewed_contours_dir": str(
            REVIEWED_CONTOURS_ROOT / "DP06" / "DP06_B251N_38kHz" / "DP06_B251N_38kHz_reviewed"
        ),
    },
    {
        "dataset_name": "DP06_B175D_38kHz",
        "cruise_name": "DP06",
        "params_file": str(EV_DVM_PARAMS_DIR / "DP06" / "params_DP06_B175D_38kHz.py"),
        "reviewed_contours_dir": str(
            REVIEWED_CONTOURS_ROOT / "DP06" / "DP06_B175D_38kHz" / "DP06_B175D_38kHz_reviewed"
        ),
    },
    {
        "dataset_name": "DP06_B082D_38kHz",
        "cruise_name": "DP06",
        "params_file": str(EV_DVM_PARAMS_DIR / "DP06" / "params_DP06_B082D_38kHz.py"),
        "reviewed_contours_dir": str(
            REVIEWED_CONTOURS_ROOT / "DP06" / "DP06_B082D_38kHz" / "DP06_B082D_38kHz_reviewed"
        ),
    },
    {
        "dataset_name": "DP06_B065N_38kHz",
        "cruise_name": "DP06",
        "params_file": str(EV_DVM_PARAMS_DIR / "DP06" / "params_DP06_B065N_38kHz.py"),
        "reviewed_contours_dir": str(
            REVIEWED_CONTOURS_ROOT / "DP06" / "DP06_B065N_38kHz" / "DP06_B065N_38kHz_reviewed"
        ),
    },
    {
        "dataset_name": "B175N_CTD246_18kHz",
        "cruise_name": "DP09",
        "params_file": str(DP09_PARAMS_DIR / "params_B175N_CTD246_18kHz.py"),
        "reviewed_contours_dir": str(
            REVIEWED_CONTOURS_ROOT / "DP09" / "B175N_CTD246" / "B175N_CTD246_reviewed"
        ),
    },
    {
        "dataset_name": "B082D_CTD255_18kHz",
        "cruise_name": "DP09",
        "params_file": str(DP09_PARAMS_DIR / "params_B082D_CTD255_18kHz.py"),
        "reviewed_contours_dir": str(
            REVIEWED_CONTOURS_ROOT / "DP09" / "B082D_CTD255" / "B082D_CTD255_reviewed"
        ),
    },
    {
        "dataset_name": "B287D_CTD253_18kHz",
        "cruise_name": "DP09",
        "params_file": str(DP09_PARAMS_DIR / "params_B287D_CTD253_18kHz.py"),
        "reviewed_contours_dir": str(
            REVIEWED_CONTOURS_ROOT / "DP09" / "B287D_CTD253_18kHz" / "B287D_CTD253_18kHz_reviewed"
        ),
    },
    {
        "dataset_name": "B082N_CTD254_18kHz",
        "cruise_name": "DP09",
        "params_file": str(DP09_PARAMS_DIR / "params_B082N_CTD254_18kHz.py"),
        "reviewed_contours_dir": str(
            REVIEWED_CONTOURS_ROOT / "DP09" / "B082N_CTD254_18kHz" / "B082N_CTD254_18kHz_reviewed"
        ),
    },
    {
        "dataset_name": "B287N_CTD252_18kHz",
        "cruise_name": "DP09",
        "params_file": str(DP09_PARAMS_DIR / "params_B287N_CTD252_18kHz.py"),
        "reviewed_contours_dir": str(
            REVIEWED_CONTOURS_ROOT / "DP09" / "B287N_CTD252_18kHz" / "B287N_CTD252_18kHz_reviewed"
        ),
    },
]
