import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def _path_from_env(var_name: str, default_relative_path: str) -> Path:
    raw_value = os.getenv(var_name, "").strip()
    candidate = Path(raw_value).expanduser() if raw_value else Path(default_relative_path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


# Root directory containing input .sv.csv files to process in batch.
INPUT_CSV_DIR = _path_from_env("ECHOGRAM_BATCH_INPUT_CSV_DIR", "EV DVM Echograms")

# Root directory where each dataset run gets its own folder.
OUTPUT_ROOT_DIR = _path_from_env("ECHOGRAM_BATCH_OUTPUT_ROOT_DIR", "Batch Results")

# Optional experiment label to group outputs by threshold trial.
# Example output path:
#   OUTPUT_ROOT_DIR / EXPERIMENT_LABEL / <dataset_name>
EXPERIMENT_LABEL = os.getenv("ECHOGRAM_BATCH_EXPERIMENT_LABEL", "ev_dvm_echograms_v1")

# Parameter source mode:
# - True: use the universal parameters in this master file for every dataset.
# - False: use per-dataset custom params files from CUSTOM_PARAMS_ROOT.
USE_MASTER_CONFIG_PARAMS = False

# Root directory containing generated per-site params, organized by cruise.
CUSTOM_PARAMS_ROOT = _path_from_env("ECHOGRAM_BATCH_PARAMS_ROOT", "EV DVM Params")

# Name of this module for runtime import by echogram_processing.
MASTER_CONFIG_MODULE = "batch_master_config"

# Dataset-specific params module naming convention:
#   params_<DATASET_ID>.py
# For file B082N_CTD254_18kHz.sv.csv, DATASET_ID is B082N_CTD254.
DATASET_PARAMS_PREFIX = "params_"

# If custom params module is missing:
# - True: fallback to MASTER_CONFIG_MODULE
# - False: skip file and report it
ALLOW_FALLBACK_TO_MASTER_CONFIG = True

# Optional filters
ONLY_FILENAMES_CONTAINING = []  # Example: ["18kHz"]
SKIP_FILENAMES_CONTAINING = []  # Example: ["38kHz"]

# Debug verbosity for batch runner console output.
VERBOSE_BATCH_DEBUG = True


# ---------------------------------------------------------------------------
# Universal processing parameters (used when USE_MASTER_CONFIG_PARAMS=True)
# ---------------------------------------------------------------------------

# In batch mode, each run sets ECHOGRAM_INPUT_FILE dynamically.
FILE_PATH = ""

# Visualization parameters (Default Echogram)
dmin = -75
dmax = None
drange = 10
USE_TIME_X_AXIS = True
TIME_AXIS_MODE = "linear"  # "true_time" or "linear"
GROUP_OUTPUTS_BY_CRUISE = True
CRUISE_NAME = None

# Optional interactive post-processing review:
# - ENABLE_LAYER_REVIEW toggles the GUI review window.
# - LAYER_REVIEW_SCOPE: "main", "all", or "choose_each_run".
# - LAYER_REVIEW_OUTPUT_SUBDIR: reviewed outputs are written to this child folder.
ENABLE_LAYER_REVIEW = False
LAYER_REVIEW_SCOPE = "choose_each_run"
LAYER_REVIEW_OUTPUT_SUBDIR = "reviewed"

# Optional reviewed contour geometry persistence/reuse:
# - SAVE_REVIEWED_CONTOURS writes reviewed contour geometry to reviewed_contours.npz.
# - LOAD_REVIEWED_CONTOURS loads an existing reviewed_contours.npz artifact.
# - SKIP_LAYER_REVIEW_IF_LOADED bypasses GUI review when a valid artifact is loaded.
# - REVIEWED_CONTOURS_SUBDIR defaults to LAYER_REVIEW_OUTPUT_SUBDIR when None.
SAVE_REVIEWED_CONTOURS = False
LOAD_REVIEWED_CONTOURS = False
SKIP_LAYER_REVIEW_IF_LOADED = True
REVIEWED_CONTOURS_FILENAME = "reviewed_contours.npz"
REVIEWED_CONTOURS_SUBDIR = None

# Analysis Ping Range
# Use None to analyze the full available ping range in each CSV.
START_PING = None
END_PING = None
START_TIME_UTC = None
END_TIME_UTC = None

# DSL/DVM Analysis Depth Range
DSL_DEPTH_START = 160
DSL_DEPTH_STOP = 500

# DVM visualization limits
DVM_VMIN = -75
DVM_VMAX = -65

# Main DSL detection thresholds
DSL_SV_THRESHOLD_MIN = -72.0
DSL_SV_THRESHOLD_MAX = -60.0

# Main DSL contour filtering and cleanup
DSL_MIN_CONTOUR_AREA = 5000
DSL_CONTOUR_EPSILON_FACTOR = 0.00005
DSL_MORPH_KERNEL_SIZE = (2, 2)
DSL_MORPH_CLOSE_ITERATIONS = 1
DSL_MORPH_OPEN_ITERATIONS = 1

# Optional wavelet preprocessing before threshold-based contour detection.
# Keep disabled by default so baseline behavior is unchanged.
ENABLE_WAVELET_PREPROCESS = False
WAVELET_NAME = "db4"
WAVELET_LEVELS = 3
WAVELET_THRESHOLD_MODE = "soft"  # "soft" or "hard"
WAVELET_THRESHOLD_SCALE = 1.0
WAVELET_CLIP_MIN = -90.0
WAVELET_CLIP_MAX = -30.0
WAVELET_APPLY_TO_SECOND_PASS = False

# --- Optional second pass (diffuse layers) ---
EXPORT_BOOLEAN_CSV = False
ENABLE_SECOND_PASS = True
SECOND_PASS_START_PING = None
SECOND_PASS_END_PING = None
SECOND_PASS_START_TIME_UTC = None
SECOND_PASS_END_TIME_UTC = None
MASK_FILL_VALUE = -90.0
PASS_1_DILATION_KERNEL_SIZE = (7, 7)
PASS_1_DILATION_ITERATIONS = 1

DIFFUSE_DSL_SV_THRESHOLD_MIN = -74.0
DIFFUSE_DSL_SV_THRESHOLD_MAX = -55.0
DIFFUSE_DSL_MIN_CONTOUR_AREA = 2000
DIFFUSE_DSL_MORPH_KERNEL_SIZE = (3, 3)
DIFFUSE_DSL_MORPH_CLOSE_ITERATIONS = 1
DIFFUSE_DSL_MORPH_OPEN_ITERATIONS = 1
DIFFUSE_DSL_CONTOUR_EPSILON_FACTOR = 0.00005
