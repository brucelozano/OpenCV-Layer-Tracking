import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
from matplotlib import dates as mdates
import os
import cv2
import importlib
import re
import json
import shutil
from sv_csv_loader import load_echoview_sv_csv, normalize_sample_column_mode
pyplot.rcParams['figure.dpi'] = 800
pyplot.rcParams['savefig.dpi'] = 800

PARAMS_FILE_PATH = os.getenv("ECHOGRAM_PARAMS_FILE")
PARAMS_MODULE_NAME = os.getenv("ECHOGRAM_PARAMS_MODULE", "params_B082D_CTD255")
if PARAMS_FILE_PATH:
    params_spec = importlib.util.spec_from_file_location("echogram_runtime_params", PARAMS_FILE_PATH)
    if params_spec is None or params_spec.loader is None:
        raise ImportError(f"Could not load params file: {PARAMS_FILE_PATH}")
    params_module = importlib.util.module_from_spec(params_spec)
    params_spec.loader.exec_module(params_module)
else:
    params_module = importlib.import_module(PARAMS_MODULE_NAME)


def get_env_bool(name, default_value):
    """
    Parse boolean environment override values in a tolerant way.
    """
    raw_value = os.getenv(name)
    if raw_value is None:
        return bool(default_value)

    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False

    print(
        f"Environment override {name}={raw_value!r} is not a valid boolean; "
        f"using default {default_value}."
    )
    return bool(default_value)


def get_env_int(name, default_value):
    """
    Parse integer environment override values.
    """
    raw_value = os.getenv(name)
    if raw_value is None:
        return int(default_value)
    try:
        return int(raw_value)
    except (TypeError, ValueError):
        print(
            f"Environment override {name}={raw_value!r} is not a valid integer; "
            f"using default {default_value}."
        )
        return int(default_value)


def get_env_float(name, default_value):
    """
    Parse float environment override values.
    """
    raw_value = os.getenv(name)
    if raw_value is None:
        return float(default_value)
    try:
        return float(raw_value)
    except (TypeError, ValueError):
        print(
            f"Environment override {name}={raw_value!r} is not a valid float; "
            f"using default {default_value}."
        )
        return float(default_value)


VALID_TIME_AXIS_MODES = {"true_time", "linear"}
VALID_WAVELET_THRESHOLD_MODES = {"soft", "hard"}


def normalize_time_axis_mode(value, context_label="TIME_AXIS_MODE"):
    """
    Normalize time-axis mode selection.

    Supported values:
      - "true_time": use per-ping timestamps for x-coordinate placement
      - "linear": use start/end timestamps with linear interpolation
    """
    normalized = str(value).strip().lower() if value is not None else "true_time"
    if normalized in VALID_TIME_AXIS_MODES:
        return normalized

    print(
        f"Warning: {context_label}={value!r} is invalid. "
        "Using 'true_time' (supported: 'true_time', 'linear')."
    )
    return "true_time"


def normalize_wavelet_threshold_mode(value, context_label="WAVELET_THRESHOLD_MODE"):
    """
    Normalize wavelet coefficient threshold mode.
    """
    normalized = str(value).strip().lower() if value is not None else "soft"
    if normalized in VALID_WAVELET_THRESHOLD_MODES:
        return normalized
    print(
        f"Warning: {context_label}={value!r} is invalid. "
        "Using 'soft' (supported: 'soft', 'hard')."
    )
    return "soft"


FILE_PATH = getattr(params_module, "FILE_PATH")
DVM_VMIN = getattr(params_module, "DVM_VMIN")
DVM_VMAX = getattr(params_module, "DVM_VMAX")
DSL_SV_THRESHOLD_MIN = getattr(params_module, "DSL_SV_THRESHOLD_MIN")
DSL_SV_THRESHOLD_MAX = getattr(params_module, "DSL_SV_THRESHOLD_MAX")
DSL_MIN_CONTOUR_AREA = getattr(params_module, "DSL_MIN_CONTOUR_AREA")
DSL_MORPH_KERNEL_SIZE = getattr(params_module, "DSL_MORPH_KERNEL_SIZE")
DSL_MORPH_CLOSE_ITERATIONS = getattr(params_module, "DSL_MORPH_CLOSE_ITERATIONS")
DSL_MORPH_OPEN_ITERATIONS = getattr(params_module, "DSL_MORPH_OPEN_ITERATIONS")
DSL_CONTOUR_EPSILON_FACTOR = getattr(params_module, "DSL_CONTOUR_EPSILON_FACTOR")
DSL_DEPTH_START = getattr(params_module, "DSL_DEPTH_START")
DSL_DEPTH_STOP = getattr(params_module, "DSL_DEPTH_STOP")
START_PING = getattr(params_module, "START_PING")
END_PING = getattr(params_module, "END_PING")
START_TIME_UTC = getattr(params_module, "START_TIME_UTC", None)
END_TIME_UTC = getattr(params_module, "END_TIME_UTC", None)
SECOND_PASS_START_PING = getattr(params_module, "SECOND_PASS_START_PING")
SECOND_PASS_END_PING = getattr(params_module, "SECOND_PASS_END_PING")
SECOND_PASS_START_TIME_UTC = getattr(params_module, "SECOND_PASS_START_TIME_UTC", None)
SECOND_PASS_END_TIME_UTC = getattr(params_module, "SECOND_PASS_END_TIME_UTC", None)
dmin = getattr(params_module, "dmin")
dmax = getattr(params_module, "dmax")
drange = getattr(params_module, "drange")
ENABLE_SECOND_PASS = getattr(params_module, "ENABLE_SECOND_PASS")
EXPORT_BOOLEAN_CSV = getattr(params_module, "EXPORT_BOOLEAN_CSV", True)
USE_TIME_X_AXIS = getattr(params_module, "USE_TIME_X_AXIS", True)
TIME_AXIS_MODE = normalize_time_axis_mode(
    getattr(params_module, "TIME_AXIS_MODE", "true_time"),
    context_label="TIME_AXIS_MODE",
)
GROUP_OUTPUTS_BY_CRUISE = getattr(params_module, "GROUP_OUTPUTS_BY_CRUISE", True)
CRUISE_NAME = getattr(params_module, "CRUISE_NAME", None)
ENABLE_LAYER_REVIEW = getattr(params_module, "ENABLE_LAYER_REVIEW", False)
LAYER_REVIEW_SCOPE = getattr(params_module, "LAYER_REVIEW_SCOPE", "choose_each_run")
LAYER_REVIEW_OUTPUT_SUBDIR = getattr(params_module, "LAYER_REVIEW_OUTPUT_SUBDIR", "reviewed")
SAVE_REVIEWED_CONTOURS = getattr(params_module, "SAVE_REVIEWED_CONTOURS", False)
LOAD_REVIEWED_CONTOURS = getattr(params_module, "LOAD_REVIEWED_CONTOURS", False)
SKIP_LAYER_REVIEW_IF_LOADED = getattr(params_module, "SKIP_LAYER_REVIEW_IF_LOADED", True)
REVIEWED_CONTOURS_FILENAME = getattr(params_module, "REVIEWED_CONTOURS_FILENAME", "reviewed_contours.npz")
REVIEWED_CONTOURS_SUBDIR = getattr(params_module, "REVIEWED_CONTOURS_SUBDIR", None)
MASK_FILL_VALUE = getattr(params_module, "MASK_FILL_VALUE")
PASS_1_DILATION_KERNEL_SIZE = getattr(params_module, "PASS_1_DILATION_KERNEL_SIZE")
PASS_1_DILATION_ITERATIONS = getattr(params_module, "PASS_1_DILATION_ITERATIONS")
DIFFUSE_DSL_SV_THRESHOLD_MIN = getattr(params_module, "DIFFUSE_DSL_SV_THRESHOLD_MIN")
DIFFUSE_DSL_SV_THRESHOLD_MAX = getattr(params_module, "DIFFUSE_DSL_SV_THRESHOLD_MAX")
DIFFUSE_DSL_MIN_CONTOUR_AREA = getattr(params_module, "DIFFUSE_DSL_MIN_CONTOUR_AREA")
DIFFUSE_DSL_MORPH_KERNEL_SIZE = getattr(params_module, "DIFFUSE_DSL_MORPH_KERNEL_SIZE")
DIFFUSE_DSL_MORPH_CLOSE_ITERATIONS = getattr(params_module, "DIFFUSE_DSL_MORPH_CLOSE_ITERATIONS")
DIFFUSE_DSL_MORPH_OPEN_ITERATIONS = getattr(params_module, "DIFFUSE_DSL_MORPH_OPEN_ITERATIONS")
DIFFUSE_DSL_CONTOUR_EPSILON_FACTOR = getattr(params_module, "DIFFUSE_DSL_CONTOUR_EPSILON_FACTOR")
ENABLE_WAVELET_PREPROCESS = getattr(params_module, "ENABLE_WAVELET_PREPROCESS", False)
WAVELET_NAME = getattr(params_module, "WAVELET_NAME", "db4")
WAVELET_LEVELS = int(getattr(params_module, "WAVELET_LEVELS", 3))
WAVELET_THRESHOLD_MODE = normalize_wavelet_threshold_mode(
    getattr(params_module, "WAVELET_THRESHOLD_MODE", "soft"),
    context_label="WAVELET_THRESHOLD_MODE",
)
WAVELET_THRESHOLD_SCALE = float(getattr(params_module, "WAVELET_THRESHOLD_SCALE", 1.0))
WAVELET_CLIP_MIN = float(getattr(params_module, "WAVELET_CLIP_MIN", -90.0))
WAVELET_CLIP_MAX = float(getattr(params_module, "WAVELET_CLIP_MAX", -30.0))
WAVELET_APPLY_TO_SECOND_PASS = getattr(params_module, "WAVELET_APPLY_TO_SECOND_PASS", False)
SAMPLE_COLUMN_MODE = normalize_sample_column_mode(
    getattr(params_module, "SAMPLE_COLUMN_MODE", "auto"),
    context_label="SAMPLE_COLUMN_MODE",
)

# Allow runtime override so batch mode can apply any params module to any input file.
FILE_PATH = os.getenv("ECHOGRAM_INPUT_FILE", FILE_PATH)
CRUISE_NAME = os.getenv("ECHOGRAM_CRUISE_NAME", CRUISE_NAME)
TIME_AXIS_MODE = normalize_time_axis_mode(
    os.getenv("ECHOGRAM_TIME_AXIS_MODE", TIME_AXIS_MODE),
    context_label="ECHOGRAM_TIME_AXIS_MODE",
)
GROUP_OUTPUTS_BY_CRUISE = get_env_bool(
    "ECHOGRAM_GROUP_OUTPUTS_BY_CRUISE",
    GROUP_OUTPUTS_BY_CRUISE,
)
ENABLE_LAYER_REVIEW = get_env_bool("ECHOGRAM_ENABLE_LAYER_REVIEW", ENABLE_LAYER_REVIEW)
SAVE_REVIEWED_CONTOURS = get_env_bool("ECHOGRAM_SAVE_REVIEWED_CONTOURS", SAVE_REVIEWED_CONTOURS)
LOAD_REVIEWED_CONTOURS = get_env_bool("ECHOGRAM_LOAD_REVIEWED_CONTOURS", LOAD_REVIEWED_CONTOURS)
SKIP_LAYER_REVIEW_IF_LOADED = get_env_bool(
    "ECHOGRAM_SKIP_LAYER_REVIEW_IF_LOADED",
    SKIP_LAYER_REVIEW_IF_LOADED,
)
REVIEWED_CONTOURS_FILENAME = os.getenv(
    "ECHOGRAM_REVIEWED_CONTOURS_FILENAME",
    REVIEWED_CONTOURS_FILENAME,
)
REVIEWED_CONTOURS_SUBDIR = os.getenv(
    "ECHOGRAM_REVIEWED_CONTOURS_SUBDIR",
    REVIEWED_CONTOURS_SUBDIR if REVIEWED_CONTOURS_SUBDIR is not None else "",
).strip() or None
REVIEWED_CONTOURS_DIR_OVERRIDE = os.getenv("ECHOGRAM_REVIEWED_CONTOURS_DIR")
ENABLE_WAVELET_PREPROCESS = get_env_bool("ECHOGRAM_WAVELET_ENABLE", ENABLE_WAVELET_PREPROCESS)
WAVELET_NAME = os.getenv("ECHOGRAM_WAVELET_NAME", WAVELET_NAME)
WAVELET_LEVELS = get_env_int("ECHOGRAM_WAVELET_LEVELS", WAVELET_LEVELS)
WAVELET_THRESHOLD_MODE = normalize_wavelet_threshold_mode(
    os.getenv("ECHOGRAM_WAVELET_THRESHOLD_MODE", WAVELET_THRESHOLD_MODE),
    context_label="ECHOGRAM_WAVELET_THRESHOLD_MODE",
)
WAVELET_THRESHOLD_SCALE = get_env_float(
    "ECHOGRAM_WAVELET_THRESHOLD_SCALE",
    WAVELET_THRESHOLD_SCALE,
)
WAVELET_CLIP_MIN = get_env_float("ECHOGRAM_WAVELET_CLIP_MIN", WAVELET_CLIP_MIN)
WAVELET_CLIP_MAX = get_env_float("ECHOGRAM_WAVELET_CLIP_MAX", WAVELET_CLIP_MAX)
WAVELET_APPLY_TO_SECOND_PASS = get_env_bool(
    "ECHOGRAM_WAVELET_APPLY_SECOND_PASS",
    WAVELET_APPLY_TO_SECOND_PASS,
)
SAMPLE_COLUMN_MODE = normalize_sample_column_mode(
    os.getenv("ECHOGRAM_SAMPLE_COLUMN_MODE", SAMPLE_COLUMN_MODE),
    context_label="ECHOGRAM_SAMPLE_COLUMN_MODE",
)

# Extract dataset name from the parameter file being used
import sys
def strip_sv_csv_suffix(filename):
    """
    Strip Echoview .sv.csv suffix; otherwise strip one extension.
    """
    lower_name = filename.lower()
    if lower_name.endswith(".sv.csv"):
        return filename[:-7]
    return os.path.splitext(filename)[0]


def get_dataset_name():
    """Extract the dataset name from the currently imported parameter file."""
    dataset_name_override = os.getenv("ECHOGRAM_DATASET_NAME")
    if dataset_name_override:
        return dataset_name_override

    # Prefer explicit module name when using standard params_<dataset>.py modules.
    params_module_name = getattr(params_module, "__name__", "")
    if params_module_name.startswith("params_"):
        return params_module_name.replace("params_", "")

    # If params were loaded by path, infer from params filename when possible.
    if PARAMS_FILE_PATH:
        params_file_stem = os.path.splitext(os.path.basename(PARAMS_FILE_PATH))[0]
        if params_file_stem.startswith("params_"):
            return params_file_stem.replace("params_", "")

    # Fallback: derive from input CSV filename.
    file_path_candidate = FILE_PATH or os.getenv("ECHOGRAM_INPUT_FILE", "")
    if file_path_candidate:
        return strip_sv_csv_suffix(os.path.basename(file_path_candidate))

    for module_name in sys.modules:
        if module_name.startswith('params_') and module_name != 'params':
            # Extract the part after 'params_'
            return module_name.replace('params_', '')
    return 'unknown_dataset'

DATASET_NAME = get_dataset_name()
print(f"Using dataset: {DATASET_NAME}")
from dsl_tracking import (preprocess_for_opencv, detect_dsl_contours, 
                      plot_echogram_with_dsl, export_dsl_layers_to_csv,
                      export_dsl_layers_to_boolean_csv, save_debug_image, get_distinct_colors,
                      wavelet_denoise_sv)
from layer_review import review_contours_interactively


def detect_cruise_name(file_path, dataset_name):
    """
    Resolve cruise name (e.g., DP08/DP09) from override or file metadata.
    """
    if CRUISE_NAME is not None:
        cruise_override = str(CRUISE_NAME).strip()
        if cruise_override:
            return cruise_override

    search_text = f"{file_path} {dataset_name}".upper()
    matches = re.findall(r"DP\d{2}", search_text)
    if matches:
        return matches[0]
    return None

def setup_figure_directory():
    """
    Resolve the output directory for figure artifacts.

    Behavior:
    - If ECHOGRAM_FIGURES_DIR is provided, treat it as the base output root.
      Cruise/dataset auto-grouping still applies under that root.
    - If ECHOGRAM_FIGURES_DIR is not provided, default root is ./Figures.
    - If cruise grouping is enabled and cruise is resolved, write to
      <root>/<CRUISE_NAME>/<DATASET_NAME>.
    - Else, write to <root>/<DATASET_NAME>.
    """
    figures_dir_override = os.getenv("ECHOGRAM_FIGURES_DIR")
    if figures_dir_override and str(figures_dir_override).strip():
        base_figures_dir = os.path.abspath(os.path.expanduser(str(figures_dir_override).strip()))
        print(f"Using custom figures root directory: {base_figures_dir}")
    else:
        base_figures_dir = os.path.join(os.getcwd(), 'Figures')

    cruise_name = detect_cruise_name(FILE_PATH, DATASET_NAME) if GROUP_OUTPUTS_BY_CRUISE else None
    if cruise_name:
        figures_dir = os.path.join(base_figures_dir, cruise_name, DATASET_NAME)
        print(f"Resolved cruise folder: {cruise_name}")
    else:
        figures_dir = os.path.join(base_figures_dir, DATASET_NAME)
        if GROUP_OUTPUTS_BY_CRUISE:
            print("Cruise folder not resolved; using dataset-only output folder.")
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)
        print(f"Created output directory at: {figures_dir}")
    return figures_dir

# Get the figures directory path
FIGURES_DIR = setup_figure_directory()

def load_and_preprocess_echogram(file_path, reset_ping_index):
    """
    Load and preprocess echogram data from a CSV file.
    
    Parameters:
    file_path (str): Path to the CSV file
    reset_ping_index (bool): Whether to reset ping indices to start from 0
    
    Returns:
    tuple: (echogram DataFrame, fixed column names, original start ping, original end ping)
    """
    # Define the names of the fixed columns Echoview creates when exporting to CSV
    fixed_column_names = [
        'Ping_index', 'Distance_gps', 'Distance_vl', 'Ping_date',
        'Ping_time', 'Ping_milliseconds', 'Latitude', 'Longitude', 
        'Depth_start', 'Depth_stop', 'Range_start', 'Range_stop', 'Sample_count'
    ]

    echogram, max_samples, resolved_sample_mode = load_echoview_sv_csv(
        file_path,
        fixed_column_names=fixed_column_names,
        sample_column_mode=SAMPLE_COLUMN_MODE,
        low_memory=False,
    )
    print(f"Maximum number of sample columns: {max_samples}")
    print(f"Sample column parser mode: {resolved_sample_mode}\n")

    # Store original start and end ping indices
    original_start_ping = echogram['Ping_index'].iloc[0]
    original_end_ping = echogram['Ping_index'].iloc[-1]
    
    # Reset ping indices to start from 0 only if requested and they don't already start from 0
    if reset_ping_index == True and original_start_ping != 0:
        echogram['Ping_index'] = echogram['Ping_index'] - original_start_ping
        print(f"Adjusted ping range: 0 to {original_end_ping - original_start_ping}")
    else:
        if not reset_ping_index:
            print("Keeping original ping indices")
        else:
            print("Ping indices already start from 0. No adjustment needed.")

    # Get placeholder value
    placeholder_val = -9.900000000000001e+37
    print("Placeholder value is: ", placeholder_val, end = "\n\n")

    # Replace placeholder_val with 0.0 (NaN causes errors when importing back to Echoview)
    echogram.replace(placeholder_val, 0.0, inplace=True)
    
    print("Replaced all values containing -9.900000000000001e+37 to 0.0")
    print(f"Original ping range: {original_start_ping} to {original_end_ping}")

    return echogram, fixed_column_names, original_start_ping, original_end_ping

# Use this to load the data
print("Loading and preprocessing echogram...\n")
echogram, fixed_column_names, original_start_ping, original_end_ping = load_and_preprocess_echogram(FILE_PATH, reset_ping_index=False)

def prepare_sample_data(echogram):
    """
    Prepare sample data from the echogram DataFrame.

    Parameters:
    echogram (pd.DataFrame): The original echogram DataFrame.

    Returns:
    tuple: (sample_data DataFrame, image_data numpy array)
    """
    # Get columns that start with 'Sample_' and exclude 'Sample_count'
    sample_columns = [col for col in echogram.columns if col.startswith('Sample_') and col != 'Sample_count']

    # Filter the DataFrame to only include these columns
    sample_data = echogram[sample_columns]
    
    # Convert the filtered DataFrame to a NumPy array
    image_data = sample_data.to_numpy(dtype='float32')
    print("Before transposing:\n")
    print("Image dtype:", image_data.dtype)
    print("Image shape:", image_data.shape, "\n")

    # Transpose the image data
    image_data = image_data.T
    print("After transposing:\n")
    print("Image dtype:", image_data.dtype)
    print("Image shape:", image_data.shape, "\n")

    return sample_data, image_data

# Example usage:
sample_data, image_data = prepare_sample_data(echogram)

# Checking the data with our new filtered DataFrame
print(sample_data.head())
print("Sample data shape: ", sample_data.shape)
print("Image data shape: ", image_data.shape)

def calculate_depth_per_sample(depth_start, depth_stop, sample_count):
    """
    Calculate the depth per sample.

    Parameters:
    depth_start (float): The starting depth in meters
    depth_stop (float): The stopping depth in meters
    sample_count (int): The number of samples

    Returns:
    float: The depth per sample in meters, rounded to 3 decimal places
    """
    depth_per_sample = (depth_stop - depth_start) / sample_count
    return round(depth_per_sample, 3)

def get_depth_parameters(dataframe):
    """
    Extract depth parameters from the first row of the dataframe.

    Parameters:
    dataframe (pd.DataFrame): The input DataFrame

    Returns:
    tuple: (depth_start, depth_stop, sample_count, total_depth)
    """
    first_row = dataframe.iloc[0]
    depth_start = first_row['Depth_start']
    depth_stop = first_row['Depth_stop']
    sample_count = first_row['Sample_count']
    total_depth = depth_stop - depth_start
    return depth_start, depth_stop, sample_count, total_depth

# Example usage:
depth_start, depth_stop, sample_count, total_depth = get_depth_parameters(echogram)
print(f"Depth start: {depth_start}")
print(f"Depth stop: {depth_stop}")
print(f"Sample count: {sample_count}")
print(f"Total depth: {total_depth}")

# Now use these values in the calculate_depth_per_sample function
depth_per_sample = calculate_depth_per_sample(depth_start, depth_stop, sample_count)
print(f"{depth_per_sample} meters per sample")

def get_dmax(dmin, dmax, drange):
    if dmax is None:
        dmax = dmin + drange  # Display maximum intensity, derived from display minimum + display range
    return dmax

dmax = get_dmax(dmin, dmax, drange) # Get the display maximum

def plot_echogram(image_data, depth_start, depth_stop, dmin, dmax, title='Echogram'):
    """
    Plot the echogram with specified parameters.

    Parameters:
    image_data (np.array): The 2D array of image data
    depth_start (float): Starting depth
    depth_stop (float): Stopping depth
    dmin (float): Minimum value for color scaling
    dmax (float): Maximum value for color scaling
    title (str): Plot title
    """
    print("Image dtype:", image_data.dtype)
    print("Image shape:", image_data.shape)
    print("Display minimum (dmin):", dmin)
    print("Display maximum (dmax):", dmax)

    use_time_axis = False
    time_axis_mode = "true_time"
    x_axis_label = 'Ping Number'
    preview_x_axis_values = None

    # echogram.png is always a full-range preview of the source CSV.
    # Time/ping analysis bounds are applied later in the DSL processing outputs.
    preview_echogram = echogram
    preview_image_data = image_data
    print("Preview image shape:", preview_image_data.shape)

    if USE_TIME_X_AXIS:
        try:
            datetime_str = (
                echogram['Ping_date'].astype(str).str.strip() + ' ' +
                echogram['Ping_time'].astype(str).str.strip()
            )
            timestamps = pd.to_datetime(datetime_str, errors='coerce')
            if timestamps.isna().any():
                raise ValueError("Failed to parse one or more ping timestamps from Ping_date/Ping_time.")

            ping_time_has_fractional_seconds = echogram['Ping_time'].astype(str).str.contains(r'\.').any()
            if 'Ping_milliseconds' in echogram.columns and not ping_time_has_fractional_seconds:
                ping_milliseconds = pd.to_numeric(
                    echogram['Ping_milliseconds'], errors='coerce'
                )
                ping_milliseconds = ping_milliseconds.where(
                    (ping_milliseconds >= 0) & (ping_milliseconds < 1000),
                    0.0,
                ).fillna(0.0)
                timestamps = timestamps + pd.to_timedelta(ping_milliseconds, unit='ms')

            preview_timestamps_utc = timestamps.dt.tz_localize('UTC')
            preview_times_naive_utc = preview_timestamps_utc.dt.tz_convert('UTC').dt.tz_localize(None)
            preview_x_axis_values = mdates.date2num(
                pd.DatetimeIndex(preview_times_naive_utc).to_pydatetime()
            )
            use_time_axis = True
            time_axis_mode = TIME_AXIS_MODE
            x_axis_label = 'Time (UTC)'
        except Exception as exc:
            print(f"Warning: failed to build time x-axis for echogram preview ({exc}). Using ping index.")
            use_time_axis = False
            preview_x_axis_values = None

    preview_start_ping = float(preview_echogram['Ping_index'].iloc[0])
    preview_end_ping = float(preview_echogram['Ping_index'].iloc[-1])
    x_extent_start = preview_start_ping
    x_extent_end = preview_end_ping
    x_axis_values = None

    if (
        use_time_axis and
        preview_x_axis_values is not None and
        len(preview_x_axis_values) == preview_image_data.shape[1]
    ):
        candidate_x_axis_values = np.asarray(preview_x_axis_values, dtype=np.float64)
        if np.all(np.isfinite(candidate_x_axis_values)) and np.all(np.diff(candidate_x_axis_values) > 0):
            x_axis_values = candidate_x_axis_values
            x_extent_start = float(x_axis_values[0])
            x_extent_end = float(x_axis_values[-1])
        else:
            print("Warning: Non-monotonic time values in echogram preview. Falling back to ping x-axis.")
            use_time_axis = False
            x_axis_label = 'Ping Number'
    elif use_time_axis:
        print("Warning: Time-axis values unavailable for echogram preview. Falling back to ping x-axis.")
        use_time_axis = False
        x_axis_label = 'Ping Number'

    plt.figure(figsize=(20, 6))
    ax = plt.gca()
    rendered_with_true_time_mesh = False
    if use_time_axis and x_axis_values is not None and time_axis_mode == "true_time":
        if preview_image_data.shape[1] == 1:
            half_step = 1.0 / 86400.0  # 1 second in Matplotlib date units
            x_edges = np.array(
                [x_axis_values[0] - half_step, x_axis_values[0] + half_step],
                dtype=np.float64,
            )
        else:
            x_edges = np.empty(preview_image_data.shape[1] + 1, dtype=np.float64)
            x_edges[1:-1] = 0.5 * (x_axis_values[:-1] + x_axis_values[1:])
            x_edges[0] = x_axis_values[0] - 0.5 * (x_axis_values[1] - x_axis_values[0])
            x_edges[-1] = x_axis_values[-1] + 0.5 * (x_axis_values[-1] - x_axis_values[-2])

        y_edges = np.linspace(
            depth_start, depth_stop, preview_image_data.shape[0] + 1, dtype=np.float64
        )
        im = ax.pcolormesh(
            x_edges,
            y_edges,
            preview_image_data,
            shading='auto',
            cmap='viridis',
            vmin=dmin,
            vmax=dmax,
        )
        ax.set_ylim(depth_stop, depth_start)
        ax.set_xlim(float(x_edges[0]), float(x_edges[-1]))
        rendered_with_true_time_mesh = True

    if not rendered_with_true_time_mesh:
        im = ax.imshow(
            preview_image_data,
            cmap='viridis',
            aspect='auto',
            extent=[x_extent_start, x_extent_end, depth_stop, depth_start],
            vmin=dmin,
            vmax=dmax,
        )

    plt.colorbar(im, label='Sv Intensity (dB)')
    plt.xlabel(x_axis_label)
    if use_time_axis and x_axis_values is not None:
        locator = mdates.AutoDateLocator(minticks=4, maxticks=10)
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        plt.setp(ax.get_xticklabels(), rotation=20, ha='right')
    plt.ylabel('Depth (m)')
    plt.title(title)
    save_path = os.path.join(FIGURES_DIR, 'echogram.png')
    plt.savefig(save_path)
    print(f"Saved echogram to: {save_path}")
    plt.close()

# Plot the echogram
plot_echogram(image_data, depth_start, depth_stop, dmin, dmax)

# If you want to specify different dmin, dmax, or title:
# plot_echogram(image_data, depth_start, depth_stop, dmin=-90, dmax=-40, title='Custom Echogram Plot')

def parse_ping_timestamps(echogram):
    """
    Parse ping timestamps and include millisecond precision when available.
    """
    required_columns = ['Ping_date', 'Ping_time']
    if not all(col in echogram.columns for col in required_columns):
        raise ValueError(f"Missing required columns: {required_columns}")

    datetime_str = (
        echogram['Ping_date'].astype(str).str.strip() + ' ' +
        echogram['Ping_time'].astype(str).str.strip()
    )
    timestamps = pd.to_datetime(datetime_str, errors='coerce')
    if timestamps.isna().any():
        raise ValueError("Failed to parse one or more ping timestamps from Ping_date/Ping_time.")

    ping_time_has_fractional_seconds = echogram['Ping_time'].astype(str).str.contains(r'\.').any()
    if 'Ping_milliseconds' in echogram.columns and not ping_time_has_fractional_seconds:
        ping_milliseconds = pd.to_numeric(
            echogram['Ping_milliseconds'], errors='coerce'
        ).fillna(0.0)
        timestamps = timestamps + pd.to_timedelta(ping_milliseconds, unit='ms')

    if timestamps.dt.tz is None:
        timestamps = timestamps.dt.tz_localize('UTC')
    else:
        timestamps = timestamps.dt.tz_convert('UTC')

    return timestamps


def create_ping_time_arrays(echogram):
    """
    Build interpolation-ready arrays of ping index and elapsed minutes.
    """
    if 'Ping_index' not in echogram.columns:
        raise ValueError("Missing required column: Ping_index")

    timestamps = parse_ping_timestamps(echogram)
    ping_values = echogram['Ping_index'].to_numpy(dtype=np.float64)
    elapsed_minutes = (timestamps - timestamps.iloc[0]).dt.total_seconds().to_numpy() / 60.0
    return ping_values, elapsed_minutes, timestamps


TIME_ONLY_UTC_PATTERN = re.compile(r"^\d{1,2}:\d{2}(?::\d{2}(?:\.\d{1,6})?)?$")


def parse_time_bound_utc(time_bound, reference_timestamps, bound_label):
    """
    Parse a time bound into a UTC timestamp.

    Accepted formats:
    - Full datetime (ISO-like), with or without timezone.
    - Time-only HH:MM[:SS[.sss]] interpreted on the same UTC date as the dataset start.
    """
    if time_bound is None:
        return None

    time_text = str(time_bound).strip()
    if not time_text:
        return None

    reference_utc = pd.DatetimeIndex(reference_timestamps).tz_convert('UTC')
    if TIME_ONLY_UTC_PATTERN.match(time_text):
        dataset_date = reference_utc[0].strftime('%Y-%m-%d')
        parse_target = f"{dataset_date} {time_text}"
    else:
        parse_target = time_text

    parsed = pd.to_datetime(parse_target, errors='coerce', utc=True)
    if pd.isna(parsed):
        raise ValueError(
            f"Could not parse {bound_label}='{time_bound}'. "
            "Use HH:MM[:SS] or an ISO datetime (e.g., 2022-08-03 11:20:00+00:00)."
        )
    return parsed


def ping_for_utc_timestamp(target_timestamp_utc, ping_values, timestamps):
    """
    Convert a UTC timestamp to the corresponding ping index by interpolation.
    """
    ping_values = np.asarray(ping_values, dtype=np.float64)
    timestamp_index = pd.DatetimeIndex(timestamps).tz_convert('UTC')
    time_seconds = timestamp_index.asi8.astype(np.float64) / 1e9
    target_seconds = target_timestamp_utc.value / 1e9

    if target_seconds <= time_seconds[0]:
        return float(ping_values[0]), "clamped_to_start"
    if target_seconds >= time_seconds[-1]:
        return float(ping_values[-1]), "clamped_to_end"

    resolved_ping = float(np.interp(target_seconds, time_seconds, ping_values))
    return resolved_ping, "within_range"


def apply_time_bounds_to_ping_range(
    pass_label,
    min_ping,
    max_ping,
    start_time_utc,
    end_time_utc,
    ping_values,
    timestamps,
):
    """
    Apply optional UTC time bounds to a ping range.
    """
    resolved_min_ping = float(min_ping)
    resolved_max_ping = float(max_ping)
    notes = []

    if start_time_utc is not None:
        resolved_min_ping, note = ping_for_utc_timestamp(start_time_utc, ping_values, timestamps)
        notes.append((f"{pass_label} start", start_time_utc, resolved_min_ping, note))

    if end_time_utc is not None:
        resolved_max_ping, note = ping_for_utc_timestamp(end_time_utc, ping_values, timestamps)
        notes.append((f"{pass_label} end", end_time_utc, resolved_max_ping, note))

    if resolved_min_ping > resolved_max_ping:
        print(
            f"Warning: {pass_label} time bounds resolve to min_ping > max_ping "
            f"({resolved_min_ping:.2f} > {resolved_max_ping:.2f}). Swapping."
        )
        resolved_min_ping, resolved_max_ping = resolved_max_ping, resolved_min_ping

    if notes:
        print(f"\n{pass_label} bounds resolved from time inputs:")
        for label, ts, ping_val, note in notes:
            ts_text = pd.DatetimeIndex([ts]).tz_convert('UTC').strftime('%Y-%m-%d %H:%M:%S')[0]
            print(f"  {label}: {ts_text} UTC -> ping {ping_val:.2f} ({note})")

    return resolved_min_ping, resolved_max_ping


def create_plot_time_axis_values(echogram):
    """
    Convert per-ping timestamps to Matplotlib date-number x-axis values.
    """
    _, _, timestamps = create_ping_time_arrays(echogram)
    timestamps_utc = pd.DatetimeIndex(timestamps).tz_convert('UTC').tz_localize(None)
    return mdates.date2num(timestamps_utc.to_pydatetime())


def validate_ping_time_alignment(ping_values, timestamps, context_label='echogram'):
    """
    Validate monotonic alignment between ping index and parsed timestamps.
    """
    ping_values = np.asarray(ping_values, dtype=np.float64)
    timestamp_index = pd.DatetimeIndex(timestamps)
    time_seconds = timestamp_index.asi8.astype(np.float64) / 1e9

    if ping_values.size != time_seconds.size:
        print(
            f"Warning [{context_label}]: ping/time length mismatch "
            f"({ping_values.size} vs {time_seconds.size})."
        )
        return

    if ping_values.size < 2:
        print(f"Ping/time validation [{context_label}]: not enough points for interval checks.")
        return

    ping_diffs = np.diff(ping_values)
    time_diffs_s = np.diff(time_seconds)
    non_increasing_ping = int(np.sum(ping_diffs <= 0))
    non_increasing_time = int(np.sum(time_diffs_s <= 0))
    duplicate_ping_count = int(ping_values.size - np.unique(ping_values).size)
    duplicate_time_count = int(time_seconds.size - np.unique(time_seconds).size)
    median_ping_step = float(np.median(ping_diffs))
    median_time_step_ms = float(np.median(time_diffs_s) * 1000.0)
    ping_time_corr = float(np.corrcoef(ping_values, time_seconds)[0, 1])

    print(f"\nPing/time alignment check [{context_label}]")
    print(f"  Points: {ping_values.size}")
    print(f"  Duplicate ping indices: {duplicate_ping_count}")
    print(f"  Duplicate timestamps: {duplicate_time_count}")
    print(f"  Non-increasing ping steps: {non_increasing_ping}")
    print(f"  Non-increasing time steps: {non_increasing_time}")
    print(f"  Median ping step: {median_ping_step:.3f}")
    print(f"  Median time step: {median_time_step_ms:.3f} ms")
    print(f"  Ping-time correlation: {ping_time_corr:.9f}")

    if non_increasing_ping > 0 or non_increasing_time > 0:
        print("  Warning: Non-monotonic ping/time sequence detected.")
    elif duplicate_ping_count > 0:
        print("  Warning: Duplicate ping indices detected.")
    else:
        print("  Status: ping and time are monotonic and aligned.\n")


def create_ping_time_mapping(echogram):
    """
    Create an explicit mapping between ping indices and timestamps.
    
    Parameters:
    echogram (pd.DataFrame): DataFrame containing 'Ping_date', 'Ping_time', and 'Ping_index' columns
    
    Returns:
    dict: Mapping of ping indices to actual timestamps (timezone-aware)
    """
    # Create explicit mapping using actual Ping_index values
    ping_values, _, timestamps = create_ping_time_arrays(echogram)
    validate_ping_time_alignment(ping_values, timestamps, context_label='full run input')
    ping_time_map = {}
    for idx in range(len(echogram)):
        ping_time_map[echogram['Ping_index'].iloc[idx]] = timestamps.iloc[idx]
    
    # Add some validation
    print(f"Created time mapping for {len(ping_time_map)} pings")
    print(f"Time range: {min(timestamps).strftime('%H:%M:%S')} to {max(timestamps).strftime('%H:%M:%S')} UTC")
    print(f"Ping range: {min(ping_time_map.keys())} to {max(ping_time_map.keys())}")
    
    return ping_time_map


def infer_expected_direction(dataset_name):
    """
    Infer expected migration direction from cast naming (e.g., B082D vs B082N).
    """
    upper_name = str(dataset_name).upper()

    # Prefer canonical cast tokens such as B082D / B175N wherever they appear.
    cast_match = re.search(r"\bB\d+[DN]\b", upper_name)
    if cast_match:
        cast_token = cast_match.group(0)
        if cast_token.endswith('D'):
            return 'downward'
        if cast_token.endswith('N'):
            return 'upward'

    # Fallback: scan underscore-delimited tokens.
    for token in upper_name.split('_'):
        if token.endswith('D') and any(ch.isdigit() for ch in token):
            return 'downward'
        if token.endswith('N') and any(ch.isdigit() for ch in token):
            return 'upward'

    return 'unknown'


def expected_direction_multiplier(expected_direction):
    """
    Convert signed depth speed (+down/-up) into expected-direction speed.
    """
    if expected_direction == 'downward':
        return 1.0
    if expected_direction == 'upward':
        return -1.0
    return 1.0


def direction_display_name(expected_direction):
    """
    Human-friendly directional label for plot text.
    """
    if expected_direction == 'downward':
        return 'Downward'
    if expected_direction == 'upward':
        return 'Upward'
    return 'Directional'


def format_utc_timestamp(timestamp, include_date=True):
    """
    Format a timestamp in UTC for display.
    """
    if timestamp is None:
        return None

    ts = pd.to_datetime(timestamp, errors='coerce', utc=True)
    if pd.isna(ts):
        return None

    if include_date:
        return ts.strftime('%Y-%m-%d %H:%M:%S')
    return ts.strftime('%H:%M:%S')


def format_time_window_label(start_timestamp, end_timestamp):
    """
    Build a compact UTC time-range label for one layer.
    """
    start_full = format_utc_timestamp(start_timestamp, include_date=True)
    end_full = format_utc_timestamp(end_timestamp, include_date=True)
    if start_full is None or end_full is None:
        return None

    start_clock = format_utc_timestamp(start_timestamp, include_date=False)
    end_clock = format_utc_timestamp(end_timestamp, include_date=False)
    same_day = start_full[:10] == end_full[:10]
    if same_day:
        return f"{start_clock} to {end_clock}"
    return f"{start_full} to {end_full}"


def resolve_layer_review_scope(scope_value):
    """
    Resolve layer review scope to one of: main, all.

    If scope is choose_each_run and stdin is interactive, prompt the user.
    Otherwise default to main.
    """
    normalized_scope = str(scope_value).strip().lower()
    valid_scopes = {"main", "all", "choose_each_run"}
    if normalized_scope not in valid_scopes:
        print(
            f"Warning: invalid LAYER_REVIEW_SCOPE='{scope_value}'. "
            "Expected main/all/choose_each_run. Defaulting to main."
        )
        normalized_scope = "main"

    if normalized_scope != "choose_each_run":
        return normalized_scope

    if sys.stdin is None or not sys.stdin.isatty():
        print("Layer review scope choose_each_run requested in non-interactive mode; defaulting to main.")
        return "main"

    print("\nLayer review scope selection")
    print("  [1] main  - review main pass contours only")
    print("  [2] all   - review main and diffuse contours")
    selected = input("Choose review scope (1/2, default=1): ").strip().lower()
    if selected in {"2", "all", "a"}:
        return "all"
    return "main"


def map_contours_between_ping_ranges(
    contours,
    source_min_ping,
    source_max_ping,
    source_image_width,
    target_min_ping,
    target_max_ping,
    target_image_width,
):
    """
    Remap contour x-coordinates between images that use different ping ranges/widths.
    """
    mapped_contours = []
    source_width_denominator = max(source_image_width - 1, 1)
    target_width_denominator = max(target_image_width - 1, 1)
    source_ping_span = source_max_ping - source_min_ping
    target_ping_span = target_max_ping - target_min_ping

    for contour in contours:
        x_coords = contour[:, :, 0].astype(np.float32)
        if source_ping_span == 0:
            ping_numbers = np.full_like(x_coords, source_min_ping, dtype=np.float32)
        else:
            ping_numbers = source_min_ping + (x_coords / source_width_denominator) * source_ping_span

        if target_ping_span == 0:
            new_x_coords = np.zeros_like(ping_numbers, dtype=np.float32)
        else:
            new_x_coords = ((ping_numbers - target_min_ping) / target_ping_span) * target_width_denominator

        remapped_contour = contour.copy()
        remapped_contour[:, :, 0] = np.clip(new_x_coords, 0, target_width_denominator)
        mapped_contours.append(remapped_contour.astype(np.int32))

    return mapped_contours


def save_layer_review_manifest(figures_dir, manifest_payload):
    """
    Persist review actions and summary to a JSON manifest.
    """
    manifest_path = os.path.join(figures_dir, "layer_review_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as manifest_file:
        json.dump(manifest_payload, manifest_file, indent=2)
    print(f"Saved layer review manifest to: {manifest_path}")
    return manifest_path


def save_reviewed_contours_status(figures_dir, status_payload):
    """
    Persist reviewed-contour load status for downstream rollout summaries.
    """
    status_path = os.path.join(figures_dir, "reviewed_contours_status.json")
    with open(status_path, "w", encoding="utf-8") as status_file:
        json.dump(status_payload, status_file, indent=2)
    print(f"Saved reviewed contour status to: {status_path}")
    return status_path


def copy_result_artifacts_to_output_folder(source_dir, output_dir, artifact_filenames):
    """
    Copy selected result artifacts into the dataset-level output folder.
    """
    copied_paths = []
    if os.path.abspath(source_dir) == os.path.abspath(output_dir):
        return copied_paths

    os.makedirs(output_dir, exist_ok=True)
    for artifact_name in artifact_filenames:
        source_path = os.path.join(source_dir, artifact_name)
        if not os.path.isfile(source_path):
            continue

        destination_path = os.path.join(output_dir, artifact_name)
        try:
            shutil.copy2(source_path, destination_path)
        except OSError as exc:
            print(
                f"Warning: failed to copy artifact '{source_path}' to "
                f"'{destination_path}': {exc}"
            )
            continue
        copied_paths.append(destination_path)

    if copied_paths:
        print(
            "Copied final artifacts into dataset output folder:\n  - "
            + "\n  - ".join(copied_paths)
        )
    return copied_paths


def summarize_review_result_for_manifest(review_result):
    """
    Keep only JSON-serializable summary fields from a review result.
    """
    if review_result is None:
        return None
    return {
        "accepted": bool(review_result.get("accepted", False)),
        "initial_count": int(review_result.get("initial_count", 0)),
        "final_count": int(review_result.get("final_count", 0)),
        "actions": review_result.get("actions", []),
    }


def resolve_layer_review_output_subdir(dataset_name, configured_subdir):
    """
    Resolve reviewed output subdir name.

    Behavior:
    - None/empty/"reviewed" -> "<dataset_name>_reviewed"
    - Custom text with "{dataset}" placeholder -> substitute dataset name
    - Any other custom text -> use as-is
    """
    if configured_subdir is None:
        return f"{dataset_name}_reviewed"

    subdir_text = str(configured_subdir).strip()
    if not subdir_text or subdir_text.lower() == "reviewed":
        return f"{dataset_name}_reviewed"

    if "{dataset}" in subdir_text:
        return subdir_text.replace("{dataset}", dataset_name)

    return subdir_text


def _optional_float(value):
    if value is None:
        return None
    try:
        float_value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(float_value):
        return None
    return float_value


def _normalize_reviewed_contours_filename(filename):
    filename_text = str(filename).strip() if filename is not None else ""
    if not filename_text:
        return "reviewed_contours.npz"
    if not filename_text.lower().endswith(".npz"):
        filename_text = f"{filename_text}.npz"
    return filename_text


def resolve_reviewed_contours_artifact_path(
    figures_dir,
    dataset_name,
    default_review_subdir,
    configured_artifact_subdir,
    configured_filename,
    directory_override=None,
):
    """
    Resolve artifact directory/file path for reviewed contour geometry persistence.
    """
    override_text = str(directory_override).strip() if directory_override is not None else ""
    if override_text:
        artifact_dir = override_text
    else:
        subdir_source = (
            configured_artifact_subdir
            if configured_artifact_subdir is not None
            else default_review_subdir
        )
        resolved_subdir = resolve_layer_review_output_subdir(dataset_name, subdir_source)
        artifact_dir = os.path.join(figures_dir, resolved_subdir)

    artifact_filename = _normalize_reviewed_contours_filename(configured_filename)
    artifact_path = os.path.join(artifact_dir, artifact_filename)
    return artifact_dir, artifact_path


def _serialize_contours_for_payload(contours):
    serialized = []
    if contours is None:
        return serialized

    for contour in contours:
        contour_array = np.asarray(contour, dtype=np.int32)
        if contour_array.size == 0:
            continue
        try:
            contour_array = contour_array.reshape(-1, 1, 2)
        except ValueError:
            print("Skipping invalid contour during serialization (cannot reshape to Nx1x2).")
            continue
        serialized.append(contour_array.tolist())
    return serialized


def _deserialize_contours_from_payload(serialized_contours):
    contours = []
    if serialized_contours is None:
        return contours

    for contour_points in serialized_contours:
        contour_array = np.asarray(contour_points, dtype=np.int32)
        if contour_array.size == 0:
            continue
        try:
            contour_array = contour_array.reshape(-1, 1, 2)
        except ValueError:
            print("Skipping invalid contour in artifact (cannot reshape to Nx1x2).")
            continue
        contours.append(contour_array.astype(np.int32))
    return contours


def build_reviewed_contours_metadata(
    dataset_name,
    scope,
    pass1_image_shape,
    min_ping,
    max_ping,
    depth_start,
    depth_stop,
    second_pass_min_ping=None,
    second_pass_max_ping=None,
    pass2_image_shape=None,
):
    """
    Build metadata used to validate reviewed contour artifacts on reload.
    """
    pass1_shape = [int(v) for v in pass1_image_shape] if pass1_image_shape is not None else None
    pass2_shape = [int(v) for v in pass2_image_shape] if pass2_image_shape is not None else None
    return {
        "dataset_name": str(dataset_name),
        "scope": str(scope) if scope is not None else None,
        "pass1_image_shape": pass1_shape,
        "pass2_image_shape": pass2_shape,
        "min_ping": _optional_float(min_ping),
        "max_ping": _optional_float(max_ping),
        "second_pass_min_ping": _optional_float(second_pass_min_ping),
        "second_pass_max_ping": _optional_float(second_pass_max_ping),
        "depth_start": _optional_float(depth_start),
        "depth_stop": _optional_float(depth_stop),
    }


def validate_reviewed_contours_metadata(artifact_metadata, expected_metadata, float_atol=1e-6):
    """
    Validate reviewed contour artifact metadata against current run context.
    """
    reasons = []
    artifact_metadata = artifact_metadata or {}
    expected_metadata = expected_metadata or {}

    if artifact_metadata.get("dataset_name") != expected_metadata.get("dataset_name"):
        reasons.append("dataset_name mismatch")

    expected_scope = expected_metadata.get("scope")
    artifact_scope = artifact_metadata.get("scope")
    if expected_scope and artifact_scope and artifact_scope != expected_scope:
        reasons.append("scope mismatch")

    artifact_pass1_shape = artifact_metadata.get("pass1_image_shape")
    expected_pass1_shape = expected_metadata.get("pass1_image_shape")
    if artifact_pass1_shape != expected_pass1_shape:
        reasons.append("pass1_image_shape mismatch")

    artifact_pass2_shape = artifact_metadata.get("pass2_image_shape")
    expected_pass2_shape = expected_metadata.get("pass2_image_shape")
    if artifact_pass2_shape != expected_pass2_shape:
        reasons.append("pass2_image_shape mismatch")

    float_keys = [
        "min_ping",
        "max_ping",
        "second_pass_min_ping",
        "second_pass_max_ping",
        "depth_start",
        "depth_stop",
    ]
    for key in float_keys:
        artifact_value = _optional_float(artifact_metadata.get(key))
        expected_value = _optional_float(expected_metadata.get(key))
        if artifact_value is None and expected_value is None:
            continue
        if artifact_value is None or expected_value is None:
            reasons.append(f"{key} mismatch")
            continue
        if not np.isclose(artifact_value, expected_value, atol=float_atol, rtol=0.0):
            reasons.append(f"{key} mismatch")

    return len(reasons) == 0, reasons


def save_reviewed_contours_artifact(
    artifact_path,
    metadata,
    main_contours,
    diffuse_contours,
    diffuse_contours_original,
):
    """
    Persist reviewed contour geometry and metadata for reproducible reruns.
    """
    artifact_payload = {
        "metadata": dict(metadata or {}),
        "main_contours": _serialize_contours_for_payload(main_contours),
        "diffuse_contours": _serialize_contours_for_payload(diffuse_contours),
        "diffuse_contours_original": _serialize_contours_for_payload(diffuse_contours_original),
    }
    artifact_payload["metadata"]["saved_utc"] = pd.Timestamp.utcnow().isoformat()

    artifact_dir = os.path.dirname(artifact_path)
    if artifact_dir:
        os.makedirs(artifact_dir, exist_ok=True)
    payload_json = json.dumps(artifact_payload, ensure_ascii=True)
    np.savez_compressed(
        artifact_path,
        payload_json=np.array([payload_json], dtype=np.str_),
    )
    print(f"Saved reviewed contour artifact to: {artifact_path}")
    return artifact_path


def load_reviewed_contours_artifact(artifact_path):
    """
    Load reviewed contour geometry artifact from disk.
    """
    if not os.path.exists(artifact_path):
        print(f"Reviewed contour artifact not found: {artifact_path}")
        return None

    try:
        with np.load(artifact_path, allow_pickle=False) as payload_npz:
            if "payload_json" not in payload_npz:
                print(f"Reviewed contour artifact missing payload_json: {artifact_path}")
                return None
            payload_json_array = payload_npz["payload_json"]
            if np.ndim(payload_json_array) == 0:
                payload_json = str(payload_json_array.tolist())
            else:
                payload_json = str(payload_json_array[0])
        payload = json.loads(payload_json)
    except Exception as exc:
        print(f"Failed to load reviewed contour artifact ({artifact_path}): {exc}")
        return None

    return {
        "metadata": payload.get("metadata", {}),
        "main_contours": _deserialize_contours_from_payload(payload.get("main_contours")),
        "diffuse_contours": _deserialize_contours_from_payload(payload.get("diffuse_contours")),
        "diffuse_contours_original": _deserialize_contours_from_payload(
            payload.get("diffuse_contours_original")
        ),
    }


WINDOWED_SPEED_MIN_DURATION_MIN = 2.0
WINDOWED_SPEED_MAX_SEGMENTS = 160
MOTION_SPEED_BASE_THRESHOLD_M_PER_MIN = 0.2
MOTION_PERSISTENCE_WINDOWS = 3
MOTION_SMOOTHING_WINDOW = 5


def extract_layer_depth_time_series(
    contour,
    image_shape,
    min_ping,
    max_ping,
    depth_start,
    depth_stop,
    ping_values,
    elapsed_minutes,
):
    """
    Extract per-layer depth/time trace from the contour mask.
    """
    img_height, img_width = image_shape
    if img_height <= 0 or img_width <= 0:
        return None
    if len(ping_values) < 2 or len(elapsed_minutes) < 2:
        return None

    ping_values = np.asarray(ping_values, dtype=np.float64)
    elapsed_minutes = np.asarray(elapsed_minutes, dtype=np.float64)

    ping_sort_idx = np.argsort(ping_values)
    ping_values_sorted = ping_values[ping_sort_idx]
    elapsed_minutes_sorted = elapsed_minutes[ping_sort_idx]

    contour_mask = np.zeros(image_shape, dtype=np.uint8)
    cv2.drawContours(contour_mask, [contour], -1, 1, thickness=cv2.FILLED)
    x_indices = np.where(contour_mask.any(axis=0))[0]
    if len(x_indices) < 2:
        return None

    ping_denominator = max(img_width - 1, 1)
    depth_denominator = max(img_height - 1, 1)

    ping_positions = min_ping + (x_indices / ping_denominator) * (max_ping - min_ping)
    ping_positions = np.clip(ping_positions, np.min(ping_values_sorted), np.max(ping_values_sorted))
    time_values = np.interp(ping_positions, ping_values_sorted, elapsed_minutes_sorted)

    profile_ping = []
    profile_time = []
    profile_depth = []
    for x_idx, ping_value, time_value in zip(x_indices, ping_positions, time_values):
        y_indices = np.where(contour_mask[:, x_idx] > 0)[0]
        if len(y_indices) == 0:
            continue

        median_y = np.median(y_indices)
        depth_value = depth_start + (median_y / depth_denominator) * (depth_stop - depth_start)
        profile_ping.append(float(ping_value))
        profile_time.append(float(time_value))
        profile_depth.append(float(depth_value))

    if len(profile_depth) < 2:
        return None

    profile_ping = np.asarray(profile_ping, dtype=np.float64)
    profile_time = np.asarray(profile_time, dtype=np.float64)
    profile_depth = np.asarray(profile_depth, dtype=np.float64)

    sort_idx = np.argsort(profile_time)
    return {
        'ping_positions': profile_ping[sort_idx],
        'time_minutes': profile_time[sort_idx],
        'depth_m': profile_depth[sort_idx],
    }


def calculate_regression_speed(time_values, depth_values):
    """
    Linear trend speed (m/min) using all depth-time samples.
    """
    finite_mask = np.isfinite(time_values) & np.isfinite(depth_values)
    if np.count_nonzero(finite_mask) < 2:
        return np.nan

    time_clean = np.asarray(time_values[finite_mask], dtype=np.float64)
    depth_clean = np.asarray(depth_values[finite_mask], dtype=np.float64)
    time_centered = time_clean - np.mean(time_clean)
    denominator = float(np.dot(time_centered, time_centered))
    if denominator <= 0:
        return np.nan

    depth_centered = depth_clean - np.mean(depth_clean)
    return float(np.dot(time_centered, depth_centered) / denominator)


def calculate_windowed_speed_summary(
    time_values,
    depth_values,
    window_duration_min=WINDOWED_SPEED_MIN_DURATION_MIN,
    max_segments=WINDOWED_SPEED_MAX_SEGMENTS,
):
    """
    Median local speed from windowed slopes plus spread (IQR).
    """
    if len(time_values) < 3 or len(depth_values) < 3:
        return np.nan, np.nan, 0

    time_values = np.asarray(time_values, dtype=np.float64)
    depth_values = np.asarray(depth_values, dtype=np.float64)
    candidate_starts = np.arange(len(time_values) - 1, dtype=np.int32)

    if len(candidate_starts) > max_segments:
        candidate_starts = np.linspace(
            0,
            len(time_values) - 2,
            num=max_segments,
            dtype=np.int32,
        )
        candidate_starts = np.unique(candidate_starts)

    local_slopes = []
    for start_idx in candidate_starts:
        target_time = time_values[start_idx] + float(window_duration_min)
        end_idx = int(np.searchsorted(time_values, target_time, side='left'))
        if end_idx <= start_idx:
            end_idx = start_idx + 1
        if end_idx >= len(time_values):
            continue

        delta_time = float(time_values[end_idx] - time_values[start_idx])
        if delta_time <= 0:
            continue
        delta_depth = float(depth_values[end_idx] - depth_values[start_idx])
        local_slopes.append(delta_depth / delta_time)

    if not local_slopes:
        delta_time = np.diff(time_values)
        delta_depth = np.diff(depth_values)
        valid = delta_time > 0
        local_slopes = (delta_depth[valid] / delta_time[valid]).tolist()

    if not local_slopes:
        return np.nan, np.nan, 0

    local_slopes = np.asarray(local_slopes, dtype=np.float64)
    q25, q75 = np.percentile(local_slopes, [25, 75])
    slope_iqr = float(q75 - q25)
    slope_median = float(np.median(local_slopes))
    return slope_median, slope_iqr, int(len(local_slopes))


def _moving_average(values, window_size):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return values

    window_size = int(max(1, min(window_size, len(values))))
    if window_size == 1:
        return values
    if window_size % 2 == 0:
        window_size = max(1, window_size - 1)
        if window_size == 1:
            return values

    kernel = np.ones(window_size, dtype=np.float64) / float(window_size)
    return np.convolve(values, kernel, mode='same')


def _first_true_run(mask, run_length):
    for idx in range(0, len(mask) - run_length + 1):
        if np.all(mask[idx:idx + run_length]):
            return idx
    return None


def _last_true_run(mask, run_length):
    for idx in range(len(mask) - run_length, -1, -1):
        if np.all(mask[idx:idx + run_length]):
            return idx
    return None


def detect_motion_window(
    time_values,
    depth_values,
    reference_speed,
    base_threshold=MOTION_SPEED_BASE_THRESHOLD_M_PER_MIN,
    persistence_windows=MOTION_PERSISTENCE_WINDOWS,
    smoothing_window=MOTION_SMOOTHING_WINDOW,
):
    """
    Detect movement onset/offset from smoothed local speed with persistence.
    """
    if len(time_values) < 3 or len(depth_values) < 3:
        return {
            'motion_detected': False,
            'motion_start_min': float(time_values[0]) if len(time_values) else np.nan,
            'motion_end_min': float(time_values[-1]) if len(time_values) else np.nan,
            'motion_threshold_m_per_min': float(base_threshold),
            'motion_confidence': 'fallback_contour_span',
            'motion_active_fraction': 0.0,
            'motion_window_count': 0,
        }

    time_values = np.asarray(time_values, dtype=np.float64)
    depth_values = np.asarray(depth_values, dtype=np.float64)

    delta_time = np.diff(time_values)
    delta_depth = np.diff(depth_values)
    valid = delta_time > 0
    if np.count_nonzero(valid) < 2:
        return {
            'motion_detected': False,
            'motion_start_min': float(time_values[0]),
            'motion_end_min': float(time_values[-1]),
            'motion_threshold_m_per_min': float(base_threshold),
            'motion_confidence': 'fallback_contour_span',
            'motion_active_fraction': 0.0,
            'motion_window_count': 0,
        }

    local_speeds = (delta_depth[valid] / delta_time[valid]).astype(np.float64)
    midpoint_times = ((time_values[:-1] + time_values[1:]) * 0.5)[valid].astype(np.float64)
    smoothed_speeds = _moving_average(local_speeds, smoothing_window)

    direction_sign = 0.0
    if np.isfinite(reference_speed) and reference_speed != 0:
        direction_sign = float(np.sign(reference_speed))
    elif depth_values[-1] != depth_values[0]:
        direction_sign = float(np.sign(depth_values[-1] - depth_values[0]))

    adaptive_threshold = float(base_threshold)
    if np.isfinite(reference_speed):
        adaptive_threshold = max(float(base_threshold), 0.25 * abs(float(reference_speed)))

    movement_mask = np.abs(smoothed_speeds) >= adaptive_threshold
    if direction_sign != 0:
        movement_mask &= np.sign(smoothed_speeds) == direction_sign

    run_length = int(max(1, min(int(persistence_windows), len(movement_mask))))
    start_idx = _first_true_run(movement_mask, run_length)
    end_run_start = _last_true_run(movement_mask, run_length)
    if start_idx is None or end_run_start is None:
        return {
            'motion_detected': False,
            'motion_start_min': float(time_values[0]),
            'motion_end_min': float(time_values[-1]),
            'motion_threshold_m_per_min': adaptive_threshold,
            'motion_confidence': 'fallback_contour_span',
            'motion_active_fraction': float(np.mean(movement_mask)) if len(movement_mask) else 0.0,
            'motion_window_count': int(len(movement_mask)),
        }

    end_idx = min(len(midpoint_times) - 1, end_run_start + run_length - 1)
    if end_idx < start_idx:
        return {
            'motion_detected': False,
            'motion_start_min': float(time_values[0]),
            'motion_end_min': float(time_values[-1]),
            'motion_threshold_m_per_min': adaptive_threshold,
            'motion_confidence': 'fallback_contour_span',
            'motion_active_fraction': float(np.mean(movement_mask)) if len(movement_mask) else 0.0,
            'motion_window_count': int(len(movement_mask)),
        }

    active_slice = movement_mask[start_idx:end_idx + 1]
    active_fraction = float(np.mean(active_slice)) if len(active_slice) else 0.0
    if active_fraction >= 0.75:
        confidence = 'high'
    elif active_fraction >= 0.55:
        confidence = 'medium'
    else:
        confidence = 'low'

    motion_start_min = float(midpoint_times[start_idx])
    motion_end_min = float(midpoint_times[end_idx])
    motion_start_min = max(motion_start_min, float(time_values[0]))
    motion_end_min = min(motion_end_min, float(time_values[-1]))

    return {
        'motion_detected': True,
        'motion_start_min': motion_start_min,
        'motion_end_min': motion_end_min,
        'motion_threshold_m_per_min': adaptive_threshold,
        'motion_confidence': confidence,
        'motion_active_fraction': active_fraction,
        'motion_window_count': int(len(movement_mask)),
    }


def _elapsed_minutes_to_iso(origin_ts, elapsed_min):
    if origin_ts is None or elapsed_min is None or not np.isfinite(elapsed_min):
        return None
    return (origin_ts + pd.to_timedelta(float(elapsed_min), unit='m')).isoformat()


def calculate_layer_vertical_metrics(
    contour,
    image_shape,
    min_ping,
    max_ping,
    depth_start,
    depth_stop,
    ping_values,
    elapsed_minutes,
    time_origin_utc=None,
):
    """
    Calculate contour-span and motion-aware vertical movement metrics for one layer.
    """
    profile = extract_layer_depth_time_series(
        contour=contour,
        image_shape=image_shape,
        min_ping=min_ping,
        max_ping=max_ping,
        depth_start=depth_start,
        depth_stop=depth_stop,
        ping_values=ping_values,
        elapsed_minutes=elapsed_minutes,
    )
    if profile is None:
        return None

    ping_positions = profile['ping_positions']
    time_values = profile['time_minutes']
    depth_values = profile['depth_m']

    # Use small edge windows for robust start/end depth estimates (current baseline method).
    edge_window = min(5, len(depth_values))
    start_depth = float(np.median(depth_values[:edge_window]))
    end_depth = float(np.median(depth_values[-edge_window:]))
    vertical_movement_m = float(end_depth - start_depth)

    contour_start_min = float(time_values[0])
    contour_end_min = float(time_values[-1])
    contour_duration_minutes = float(contour_end_min - contour_start_min)
    if contour_duration_minutes <= 0:
        speed_m_per_min = np.nan
    else:
        speed_m_per_min = float(vertical_movement_m / contour_duration_minutes)

    regression_speed = calculate_regression_speed(time_values, depth_values)
    windowed_speed, windowed_speed_iqr, windowed_speed_count = calculate_windowed_speed_summary(
        time_values=time_values,
        depth_values=depth_values,
    )

    reference_motion_speed = regression_speed if np.isfinite(regression_speed) else speed_m_per_min
    motion_window = detect_motion_window(
        time_values=time_values,
        depth_values=depth_values,
        reference_speed=reference_motion_speed,
    )

    motion_start_min = float(motion_window['motion_start_min'])
    motion_end_min = float(motion_window['motion_end_min'])
    motion_duration_min = float(max(0.0, motion_end_min - motion_start_min))

    start_ping = float(np.min(ping_positions))
    end_ping = float(np.max(ping_positions))

    ping_values = np.asarray(ping_values, dtype=np.float64)
    elapsed_minutes = np.asarray(elapsed_minutes, dtype=np.float64)
    time_sort_idx = np.argsort(elapsed_minutes)
    elapsed_minutes_sorted = elapsed_minutes[time_sort_idx]
    ping_by_time_sorted = ping_values[time_sort_idx]
    motion_start_ping = float(
        np.interp(
            motion_start_min,
            elapsed_minutes_sorted,
            ping_by_time_sorted,
            left=ping_by_time_sorted[0],
            right=ping_by_time_sorted[-1],
        )
    )
    motion_end_ping = float(
        np.interp(
            motion_end_min,
            elapsed_minutes_sorted,
            ping_by_time_sorted,
            left=ping_by_time_sorted[0],
            right=ping_by_time_sorted[-1],
        )
    )

    origin_ts = None
    if time_origin_utc is not None:
        parsed_origin = pd.to_datetime(time_origin_utc, errors='coerce', utc=True)
        if not pd.isna(parsed_origin):
            origin_ts = parsed_origin

    contour_start_time_utc = _elapsed_minutes_to_iso(origin_ts, contour_start_min)
    contour_end_time_utc = _elapsed_minutes_to_iso(origin_ts, contour_end_min)
    motion_start_time_utc = _elapsed_minutes_to_iso(origin_ts, motion_start_min)
    motion_end_time_utc = _elapsed_minutes_to_iso(origin_ts, motion_end_min)

    return {
        # Backward-compatible contour-span fields.
        'start_ping': start_ping,
        'end_ping': end_ping,
        'start_time_utc': contour_start_time_utc,
        'end_time_utc': contour_end_time_utc,
        'start_depth_m': start_depth,
        'end_depth_m': end_depth,
        'vertical_movement_m': vertical_movement_m,
        'duration_min': contour_duration_minutes,
        'speed_m_per_min': float(speed_m_per_min),
        # Explicit contour-span aliases.
        'contour_start_ping': start_ping,
        'contour_end_ping': end_ping,
        'contour_start_time_utc': contour_start_time_utc,
        'contour_end_time_utc': contour_end_time_utc,
        'contour_duration_min': contour_duration_minutes,
        'contour_start_depth_m': start_depth,
        'contour_end_depth_m': end_depth,
        'speed_m_per_min_endpoint': float(speed_m_per_min),
        # Additional speed metrics.
        'speed_m_per_min_regression': float(regression_speed) if np.isfinite(regression_speed) else np.nan,
        'speed_m_per_min_windowed': float(windowed_speed) if np.isfinite(windowed_speed) else np.nan,
        'speed_m_per_min_windowed_iqr': float(windowed_speed_iqr) if np.isfinite(windowed_speed_iqr) else np.nan,
        'speed_m_per_min_windowed_sample_count': int(windowed_speed_count),
        # Motion onset/offset metrics.
        'motion_detected': bool(motion_window['motion_detected']),
        'motion_confidence': motion_window['motion_confidence'],
        'motion_threshold_m_per_min': float(motion_window['motion_threshold_m_per_min']),
        'motion_active_fraction': float(motion_window['motion_active_fraction']),
        'motion_window_count': int(motion_window['motion_window_count']),
        'motion_start_min': motion_start_min,
        'motion_end_min': motion_end_min,
        'motion_duration_min': motion_duration_min,
        'motion_start_ping': motion_start_ping,
        'motion_end_ping': motion_end_ping,
        'motion_start_time_utc': motion_start_time_utc,
        'motion_end_time_utc': motion_end_time_utc,
    }


def _resolve_speed_plot_key(layer_metrics, requested_key):
    candidate_keys = [
        requested_key,
        'primary_speed_m_per_min_display',
        'speed_m_per_min_regression_display',
        'speed_m_per_min_windowed_display',
        'speed_m_per_min_display',
    ]
    for key in candidate_keys:
        if any(_optional_float(metric.get(key)) is not None for metric in layer_metrics):
            return key
    return requested_key


def _speed_label_from_key(speed_key):
    if speed_key == 'speed_m_per_min_regression_display':
        return 'trend speed'
    if speed_key == 'speed_m_per_min_windowed_display':
        return 'windowed speed'
    if speed_key == 'primary_speed_m_per_min_display':
        return 'primary speed'
    return 'speed'


def _direction_flag_key_for_speed(speed_key, override=None):
    if override:
        return override

    key_mapping = {
        'primary_speed_m_per_min_display': 'primary_opposes_expected_direction',
        'speed_m_per_min_regression_display': 'opposes_expected_direction_regression',
        'speed_m_per_min_windowed_display': 'opposes_expected_direction_windowed',
        'speed_m_per_min_display': 'opposes_expected_direction',
    }
    return key_mapping.get(speed_key, 'opposes_expected_direction')


def add_expected_direction_speed_fields(layer_metrics, direction_multiplier):
    """
    Add signed/expected/display variants for each speed metric.
    """
    speed_sources = [
        ('speed_m_per_min', ''),
        ('speed_m_per_min_regression', 'regression'),
        ('speed_m_per_min_windowed', 'windowed'),
    ]

    for source_key, suffix in speed_sources:
        raw_speed = layer_metrics.get(source_key, np.nan)
        signed_value = _optional_float(raw_speed)
        signed_speed = signed_value if signed_value is not None else np.nan
        expected_speed = signed_speed * direction_multiplier if np.isfinite(signed_speed) else np.nan
        display_speed = abs(expected_speed) if np.isfinite(expected_speed) else np.nan

        layer_metrics[f'{source_key}_signed'] = signed_speed
        layer_metrics[f'{source_key}_expected'] = expected_speed
        layer_metrics[f'{source_key}_display'] = display_speed

        direction_flag_key = 'opposes_expected_direction' if not suffix else f'opposes_expected_direction_{suffix}'
        layer_metrics[direction_flag_key] = bool(np.isfinite(expected_speed) and expected_speed < 0)

    # Preserve legacy cast-direction flag semantics for endpoint speed.
    layer_metrics['opposes_cast_direction'] = bool(layer_metrics.get('opposes_expected_direction', False))

    primary_candidates = [
        ('speed_m_per_min_regression', 'regression'),
        ('speed_m_per_min_windowed', 'windowed'),
        ('speed_m_per_min', 'endpoint'),
    ]
    selected_source = 'speed_m_per_min'
    selected_method = 'endpoint'
    for source_key, method_name in primary_candidates:
        if _optional_float(layer_metrics.get(f'{source_key}_display')) is not None:
            selected_source = source_key
            selected_method = method_name
            break

    layer_metrics['primary_speed_metric'] = selected_method
    layer_metrics['primary_speed_m_per_min_signed'] = layer_metrics.get(f'{selected_source}_signed', np.nan)
    layer_metrics['primary_speed_m_per_min_expected'] = layer_metrics.get(f'{selected_source}_expected', np.nan)
    layer_metrics['primary_speed_m_per_min_display'] = layer_metrics.get(f'{selected_source}_display', np.nan)

    if selected_method == 'regression':
        primary_flag = layer_metrics.get('opposes_expected_direction_regression', False)
    elif selected_method == 'windowed':
        primary_flag = layer_metrics.get('opposes_expected_direction_windowed', False)
    else:
        primary_flag = layer_metrics.get('opposes_expected_direction', False)
    layer_metrics['primary_opposes_expected_direction'] = bool(primary_flag)


def _metric_start_sort_key(metric):
    """
    Build a stable sort key for ordering layers by earliest contour start time.
    """
    for time_key in ['contour_start_time_utc', 'start_time_utc', 'motion_start_time_utc']:
        raw_value = metric.get(time_key)
        if raw_value is None or raw_value == "":
            continue
        parsed_ts = pd.to_datetime(raw_value, errors='coerce', utc=True)
        if pd.notna(parsed_ts):
            return 0, int(parsed_ts.value)
    return 1, 0


def build_time_ranked_main_layer_outputs(main_layer_metrics, main_contours):
    """
    Return (time-ranked metrics, time-ranked contours, colors) for main layers.
    """
    if not main_layer_metrics or not main_contours:
        return [], [], []

    sortable_entries = []
    for fallback_index, metric in enumerate(main_layer_metrics):
        contour_index_raw = metric.get('_contour_index', fallback_index)
        try:
            contour_index = int(contour_index_raw)
        except (TypeError, ValueError):
            contour_index = fallback_index
        sortable_entries.append(
            {
                'metric': dict(metric),
                'contour_index': contour_index,
                'fallback_index': fallback_index,
                'sort_key': _metric_start_sort_key(metric),
            }
        )

    sortable_entries.sort(
        key=lambda entry: (
            entry['sort_key'][0],
            entry['sort_key'][1],
            entry['fallback_index'],
        )
    )

    ranked_metrics = []
    ranked_contours = []
    ranked_colors = []
    for entry in sortable_entries:
        contour_index = entry['contour_index']
        if contour_index < 0 or contour_index >= len(main_contours):
            continue
        ranked_metrics.append(entry['metric'])
        ranked_contours.append(main_contours[contour_index])

    if not ranked_metrics or not ranked_contours:
        return [], [], []

    ranked_colors = get_distinct_colors(len(ranked_metrics))
    for rank, (metric, color) in enumerate(zip(ranked_metrics, ranked_colors), start=1):
        metric['label'] = f'Main Layer {rank}'
        metric['color'] = color
        metric.pop('_contour_index', None)

    return ranked_metrics, ranked_contours, ranked_colors


def save_layer_speed_figure(
    echogram_image_path,
    layer_metrics,
    output_path,
    dataset_name,
    expected_direction,
    plot_title,
    speed_key='primary_speed_m_per_min_display',
    ylabel=None,
    direction_flag_key=None,
):
    """
    Save a two-panel figure with the DSL echogram (top) and layer speeds (bottom).
    """
    speed_key = _resolve_speed_plot_key(layer_metrics, speed_key)
    direction_flag_key = _direction_flag_key_for_speed(speed_key, direction_flag_key)
    valid_metrics = [metric for metric in layer_metrics if _optional_float(metric.get(speed_key)) is not None]
    if not valid_metrics:
        print("No valid layer speed metrics available to plot.")
        return

    layer_count = len(valid_metrics)
    if layer_count <= 3:
        fig_size = (16, 9)
        height_ratios = [3, 1.25]
        bar_width = 0.42
    elif layer_count <= 6:
        fig_size = (17, 10)
        height_ratios = [3, 1.5]
        bar_width = 0.55
    else:
        fig_size = (18, 12)
        height_ratios = [3, 2]
        bar_width = 0.7

    if ylabel is None:
        metric_label = _speed_label_from_key(speed_key)
        ylabel = f"{direction_display_name(expected_direction)} {metric_label} (m/min)"

    figure, (echogram_ax, speed_ax) = plt.subplots(
        2, 1, figsize=fig_size, gridspec_kw={'height_ratios': height_ratios}
    )

    echogram_img = plt.imread(echogram_image_path)
    echogram_ax.imshow(echogram_img)
    echogram_ax.axis('off')
    echogram_ax.set_title(f'{dataset_name} Layer Regions')

    labels = [metric['label'] for metric in valid_metrics]
    speeds = [float(metric[speed_key]) for metric in valid_metrics]
    colors = [metric['color'] for metric in valid_metrics]

    x_positions = np.arange(len(valid_metrics))
    bars = speed_ax.bar(
        x_positions,
        speeds,
        width=bar_width,
        color=colors,
        edgecolor='black',
        linewidth=0.8,
        alpha=0.85,
    )
    speed_ax.axhline(0, color='black', linewidth=1.0)
    speed_ax.set_xticks(x_positions)
    speed_ax.set_xticklabels(labels, rotation=25, ha='right')
    speed_ax.set_xlabel('Layer / Filament')
    speed_ax.set_ylabel(ylabel)
    speed_ax.grid(True, axis='y', alpha=0.25)
    speed_ax.set_xlim(-0.5, layer_count - 0.5)

    max_abs_speed = max(abs(speed) for speed in speeds) if speeds else 0.0
    label_offset = max(0.1, max_abs_speed * 0.06)

    # Reserve headroom/footroom so numeric labels never clip at axis bounds.
    min_speed = min(speeds) if speeds else 0.0
    max_speed = max(speeds) if speeds else 0.0
    speed_span = max_speed - min_speed
    extra_margin = max(0.2, max_abs_speed * 0.1, speed_span * 0.12)

    if min_speed >= 0:
        y_min = 0.0
        y_max = max_speed + label_offset + extra_margin
    elif max_speed <= 0:
        y_min = min_speed - label_offset - extra_margin
        y_max = 0.0
    else:
        y_min = min_speed - label_offset - extra_margin
        y_max = max_speed + label_offset + extra_margin

    if np.isclose(y_min, y_max):
        y_max = y_min + 1.0
    speed_ax.set_ylim(y_min, y_max)

    opposite_direction_count = 0
    for bar, speed, metric in zip(bars, speeds, valid_metrics):
        if metric.get(direction_flag_key, False):
            bar.set_edgecolor('red')
            bar.set_linewidth(1.8)
            opposite_direction_count += 1
        text_y = speed + label_offset if speed >= 0 else speed - label_offset
        va = 'bottom' if speed >= 0 else 'top'
        speed_ax.text(
            bar.get_x() + (bar.get_width() / 2.0),
            text_y,
            f'{speed:.2f}',
            ha='center',
            va=va,
            fontsize=9,
        )

    speed_ax.set_title(plot_title)
    time_window_lines = []
    for metric in valid_metrics:
        layer_window_label = format_time_window_label(
            metric.get('contour_start_time_utc', metric.get('start_time_utc')),
            metric.get('contour_end_time_utc', metric.get('end_time_utc')),
        )
        if layer_window_label:
            time_window_lines.append(f"{metric['label']}: {layer_window_label}")

    if time_window_lines:
        max_time_lines = 8
        displayed_time_lines = time_window_lines[:max_time_lines]
        if len(time_window_lines) > max_time_lines:
            remaining = len(time_window_lines) - max_time_lines
            displayed_time_lines.append(f"... (+{remaining} more)")

        speed_ax.text(
            0.01, 0.98,
            "Layer start/end (UTC)\n" + "\n".join(displayed_time_lines),
            transform=speed_ax.transAxes,
            ha='left',
            va='top',
            fontsize=8,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85),
        )

    direction_text = (
        f"Red outlines: {opposite_direction_count} layer(s) moved opposite expected "
        f"{direction_display_name(expected_direction).lower()} direction."
        if opposite_direction_count > 0
        else "All displayed layers moved in the expected cast direction."
    )
    speed_ax.text(
        0.99, 0.98,
        direction_text,
        transform=speed_ax.transAxes,
        ha='right',
        va='top',
        fontsize=8,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85),
    )

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close(figure)
    print(f"Saved layer speed summary figure to: {output_path}")

def visualize_echogram(image_data, depth_start, depth_stop, min_ping, max_ping,
                      original_data_start_depth=None, original_data_total_depth_span=None,
                      vmin=-75, vmax=-65, save_path=None, title='Echogram'):
    """
    Visualize the echogram with correct ping indices and depth range.
    
    Args:
        image_data: The image data array
        depth_start: Starting depth for display
        depth_stop: Stopping depth for display
        min_ping: Minimum ping index
        max_ping: Maximum ping index
        original_data_start_depth: The absolute depth of image_data[0,:] (e.g., -0.06m)
        original_data_total_depth_span: The depth span that image_data.shape[0] represents
        vmin: Minimum value for color scaling
        vmax: Maximum value for color scaling
        save_path: Path to save the figure
        title: Plot title
    """
    plt.figure(figsize=(15, 8))
    
    # If original depth parameters aren't provided, try to get them from the global echogram
    if original_data_start_depth is None or original_data_total_depth_span is None:
        original_data_start_depth = echogram.iloc[0]['Depth_start']
        original_data_total_depth_span = echogram.iloc[0]['Depth_stop'] - echogram.iloc[0]['Depth_start']
    
    # Calculate indices for cropping the visualization by depth
    if original_data_total_depth_span <= 0:
        print(f"Warning: Invalid total depth span ({original_data_total_depth_span}) in visualize_echogram. Skipping depth cropping.")
        cropped_image = image_data
    else:
        pixels_per_meter = image_data.shape[0] / original_data_total_depth_span
        
        # Calculate crop indices relative to the start of image_data
        depth_idx_start = int((depth_start - original_data_start_depth) * pixels_per_meter)
        depth_idx_end = int((depth_stop - original_data_start_depth) * pixels_per_meter)
        
        depth_idx_start = max(0, depth_idx_start)
        depth_idx_end = min(image_data.shape[0], depth_idx_end)
        
        if depth_idx_start >= depth_idx_end:
            print(f"Warning: Invalid depth crop indices ({depth_idx_start} to {depth_idx_end}). Plotting full depth.")
            cropped_image = image_data
        else:
            cropped_image = image_data[depth_idx_start:depth_idx_end, :]
    
    # Plot echogram with correct ping range
    im = plt.imshow(cropped_image, aspect='auto', cmap='viridis',
                   extent=[min_ping, max_ping, depth_stop, depth_start],
                   vmin=vmin, vmax=vmax)
    
    cbar = plt.colorbar(im)
    cbar.set_label('Sv (dB)')
    
    plt.xlabel('Ping Number')
    plt.ylabel('Depth (m)')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved echogram to: {save_path}")
    
    plt.close()

def process_echogram_with_dsl_detection(start_ping=None, end_ping=None, show_outline=True):
    """Process the echogram and detect DSL using a two-pass approach with potentially different ping ranges."""
    print("\nStarting echogram processing pipeline...")
    
    # Get the actual ping range from the echogram
    actual_start_ping = echogram['Ping_index'].iloc[0]
    actual_end_ping = echogram['Ping_index'].iloc[-1]
    print(f"Available ping range in data: {actual_start_ping} to {actual_end_ping}")

    full_ping_values, _, full_timestamps = create_ping_time_arrays(echogram)
    
    # Validate and adjust ping ranges for first pass
    min_ping = start_ping if start_ping is not None else actual_start_ping
    max_ping = end_ping if end_ping is not None else actual_end_ping

    first_pass_start_time_utc = parse_time_bound_utc(START_TIME_UTC, full_timestamps, "START_TIME_UTC")
    first_pass_end_time_utc = parse_time_bound_utc(END_TIME_UTC, full_timestamps, "END_TIME_UTC")
    if first_pass_start_time_utc is not None or first_pass_end_time_utc is not None:
        min_ping, max_ping = apply_time_bounds_to_ping_range(
            pass_label="Pass 1",
            min_ping=min_ping,
            max_ping=max_ping,
            start_time_utc=first_pass_start_time_utc,
            end_time_utc=first_pass_end_time_utc,
            ping_values=full_ping_values,
            timestamps=full_timestamps,
        )

    # Validate and adjust ping ranges for second pass
    second_pass_min_ping = SECOND_PASS_START_PING if SECOND_PASS_START_PING is not None else min_ping
    second_pass_max_ping = SECOND_PASS_END_PING if SECOND_PASS_END_PING is not None else max_ping

    second_pass_start_time_utc = parse_time_bound_utc(
        SECOND_PASS_START_TIME_UTC,
        full_timestamps,
        "SECOND_PASS_START_TIME_UTC",
    )
    second_pass_end_time_utc = parse_time_bound_utc(
        SECOND_PASS_END_TIME_UTC,
        full_timestamps,
        "SECOND_PASS_END_TIME_UTC",
    )
    if second_pass_start_time_utc is not None or second_pass_end_time_utc is not None:
        second_pass_min_ping, second_pass_max_ping = apply_time_bounds_to_ping_range(
            pass_label="Pass 2",
            min_ping=second_pass_min_ping,
            max_ping=second_pass_max_ping,
            start_time_utc=second_pass_start_time_utc,
            end_time_utc=second_pass_end_time_utc,
            ping_values=full_ping_values,
            timestamps=full_timestamps,
        )
    
    # Validate both ping ranges
    for pass_num, (pass_min, pass_max) in enumerate([(min_ping, max_ping), 
                                                    (second_pass_min_ping, second_pass_max_ping)], 1):
        if pass_min > actual_end_ping or pass_max < actual_start_ping:
            print(f"Warning: Pass {pass_num} ping range ({pass_min} to {pass_max}) is outside available range")
            if pass_num == 1:
                min_ping = actual_start_ping
                max_ping = actual_end_ping
            else:
                second_pass_min_ping = min_ping  # Default to first pass range
                second_pass_max_ping = max_ping
        else:
            # Adjust individual out-of-range values
            if pass_min < actual_start_ping:
                if pass_num == 1:
                    min_ping = actual_start_ping
                else:
                    second_pass_min_ping = actual_start_ping
            if pass_max > actual_end_ping:
                if pass_num == 1:
                    max_ping = actual_end_ping
                else:
                    second_pass_max_ping = actual_end_ping
    
    print(f"Using first pass ping range: {min_ping} to {max_ping}")
    print(f"Using second pass ping range: {second_pass_min_ping} to {second_pass_max_ping}")
    if USE_TIME_X_AXIS:
        print(f"Primary x-axis mode: Time (UTC) [{TIME_AXIS_MODE}]")
    else:
        print("Primary x-axis mode: Ping index")
    if ENABLE_WAVELET_PREPROCESS:
        print(
            "Wavelet preprocessing enabled: "
            f"name={WAVELET_NAME}, levels={WAVELET_LEVELS}, "
            f"mode={WAVELET_THRESHOLD_MODE}, scale={WAVELET_THRESHOLD_SCALE:.3f}, "
            f"clip=[{WAVELET_CLIP_MIN:.1f}, {WAVELET_CLIP_MAX:.1f}], "
            f"pass2={WAVELET_APPLY_TO_SECOND_PASS}"
        )
    
    # Get depth parameters first
    print("\nGetting depth parameters...")
    original_depth_start, original_depth_stop, sample_count, original_total_depth = get_depth_parameters(echogram)
    print(f"Full depth range: {original_depth_start:.1f}m to {original_depth_stop:.1f}m")
    
    # Override depth range for DSL detection if needed
    depth_start = DSL_DEPTH_START  # Set desired start depth
    depth_stop = DSL_DEPTH_STOP   # Set desired stop depth
    print(f"Processing depth range: {depth_start:.1f}m to {depth_stop:.1f}m")
    
    # Crop the data for first pass analysis
    ping_mask = (echogram['Ping_index'] >= min_ping) & (echogram['Ping_index'] <= max_ping)
    echogram_cropped = echogram[ping_mask]
    _, image_data_for_pings = prepare_sample_data(echogram_cropped)
    main_plot_x_values = create_plot_time_axis_values(echogram_cropped) if USE_TIME_X_AXIS else None
    
    # Calculate depth indices for cropping
    pixels_per_meter = image_data_for_pings.shape[0] / original_total_depth
    depth_idx_start = int((depth_start - original_depth_start) * pixels_per_meter)
    depth_idx_end = int((depth_stop - original_depth_start) * pixels_per_meter)
    
    # Ensure indices are within bounds
    depth_idx_start = max(0, depth_idx_start)
    depth_idx_end = min(image_data_for_pings.shape[0], depth_idx_end)
    
    # Crop image data by depth for first pass DSL detection
    cropped_image_data_for_dsl = image_data_for_pings[depth_idx_start:depth_idx_end, :]
    
    if ENABLE_WAVELET_PREPROCESS:
        print("Applying wavelet denoising before Pass 1 detection...")
        cropped_image_data_for_dsl = wavelet_denoise_sv(
            cropped_image_data_for_dsl,
            wavelet_name=WAVELET_NAME,
            levels=WAVELET_LEVELS,
            threshold_mode=WAVELET_THRESHOLD_MODE,
            threshold_scale=WAVELET_THRESHOLD_SCALE,
            clip_min=WAVELET_CLIP_MIN,
            clip_max=WAVELET_CLIP_MAX,
        )

    # Check if we're working with resampled data and enhance if needed
    is_resampled_data = "resampled" in DATASET_NAME.lower() or "7x7" in DATASET_NAME.lower() or "3x3" in DATASET_NAME.lower()
    if is_resampled_data:
        print("Detected resampled data - applying enhancement...")
        from dsl_tracking import enhance_resampled_data
        sharpen_edges = not ENABLE_WAVELET_PREPROCESS
        if ENABLE_WAVELET_PREPROCESS:
            print(
                "Wavelet preprocessing is enabled, so resampled-data sharpening is disabled "
                "to avoid reintroducing high-frequency noise."
            )
        cropped_image_data_for_dsl = enhance_resampled_data(
            cropped_image_data_for_dsl,
            DSL_SV_THRESHOLD_MIN,
            DSL_SV_THRESHOLD_MAX,
            sharpen_edges=sharpen_edges,
        )
    
    # Create ping time mapping with millisecond precision.
    print("\nCreating ping time mapping...")
    ping_time_map = create_ping_time_mapping(echogram)
    
    # --- PASS 1: Detect Main DSL Layers ---
    print("\n--- Starting Pass 1: Main DSL Detection ---")
    dsl_debug_dir_pass1 = os.path.join(FIGURES_DIR, "dsl_debug_pass1")
    if not os.path.exists(dsl_debug_dir_pass1):
        os.makedirs(dsl_debug_dir_pass1)
    
    main_dsl_contours, main_dsl_mask = detect_dsl_contours(
        None,
        cropped_image_data_for_dsl,
        sv_threshold_min=DSL_SV_THRESHOLD_MIN,
        sv_threshold_max=DSL_SV_THRESHOLD_MAX,
        min_contour_area=DSL_MIN_CONTOUR_AREA,
        morph_kernel_size=DSL_MORPH_KERNEL_SIZE,
        morph_close_iterations=DSL_MORPH_CLOSE_ITERATIONS,
        morph_open_iterations=DSL_MORPH_OPEN_ITERATIONS,
        contour_epsilon_factor=DSL_CONTOUR_EPSILON_FACTOR,
        debug_dvm_vmin=DVM_VMIN,
        debug_dvm_vmax=DVM_VMAX,
        figures_dir_for_debug=dsl_debug_dir_pass1,
        stage_prefix="pass1_"
    )
    print(f"Pass 1: Found {len(main_dsl_contours)} main DSL contours.")
    
    # Initialize list to store all contours
    all_detected_contours = list(main_dsl_contours)
    layer_review_applied = False
    layer_review_scope = None
    layer_review_output_subdir = resolve_layer_review_output_subdir(
        DATASET_NAME,
        LAYER_REVIEW_OUTPUT_SUBDIR,
    )
    review_output_figures_dir = FIGURES_DIR
    review_manifest = None
    reviewed_contours_loaded = False
    reviewed_contours_load_metadata = None
    reviewed_contours_artifact_saved_path = None
    reviewed_contours_artifact_dir, reviewed_contours_artifact_path = resolve_reviewed_contours_artifact_path(
        figures_dir=FIGURES_DIR,
        dataset_name=DATASET_NAME,
        default_review_subdir=LAYER_REVIEW_OUTPUT_SUBDIR,
        configured_artifact_subdir=REVIEWED_CONTOURS_SUBDIR,
        configured_filename=REVIEWED_CONTOURS_FILENAME,
        directory_override=REVIEWED_CONTOURS_DIR_OVERRIDE,
    )
    reviewed_contours_status = {
        "dataset_name": DATASET_NAME,
        "requested_load": bool(LOAD_REVIEWED_CONTOURS),
        "loaded_from_artifact": False,
        "load_status": "requested" if LOAD_REVIEWED_CONTOURS else "not_requested",
        "load_reason": None,
        "artifact_path": reviewed_contours_artifact_path,
        "metadata_mismatch_reasons": [],
    }
    
    # Plot echogram with ALL DVM contours, color-coded by type
    # 1. Plot Pass 1 (Main) layers with distinct colors
    if main_dsl_contours:
        main_dsl_path = 'echogram_with_main_dsl.png'
        plot_echogram_with_dsl(
            image_data=cropped_image_data_for_dsl,
            depth_start=depth_start,
            depth_stop=depth_stop,
            min_ping=min_ping,
            max_ping=max_ping,
            contours=main_dsl_contours,  # Just pass main contours
            vmin=DVM_VMIN,
            vmax=DVM_VMAX,
            title=f'{DATASET_NAME} Main DVM Layers ({min_ping}-{max_ping})',
            save_path=main_dsl_path,
            figures_dir=FIGURES_DIR,
            show_outline=show_outline,
            contour_info=None,  # Use default coloring for distinct colors
            x_axis_values=main_plot_x_values,
            x_axis_is_time=USE_TIME_X_AXIS,
            x_axis_label='Time (UTC)' if USE_TIME_X_AXIS else 'Ping Number',
            x_axis_mode=TIME_AXIS_MODE,
        )
        print(f"Saved main DSL layer visualization")

    # 2. Process and plot Pass 2 (Diffuse) layers
    diffuse_dsl_contours = []
    diffuse_dsl_contours_original = []  # Initialize for CSV export
    second_pass_echogram = None
    diffuse_plot_x_values = None
    modified_sv_data = None
    if ENABLE_SECOND_PASS and main_dsl_contours:
        print("\n--- Starting Pass 2: Diffuse DSL Detection ---")
        
        # Crop the data for second pass analysis using second pass parameters
        second_pass_ping_mask = (echogram['Ping_index'] >= second_pass_min_ping) & \
                               (echogram['Ping_index'] <= second_pass_max_ping)
        second_pass_echogram = echogram[second_pass_ping_mask]
        _, second_pass_image_data = prepare_sample_data(second_pass_echogram)
        diffuse_plot_x_values = create_plot_time_axis_values(second_pass_echogram) if USE_TIME_X_AXIS else None
        
        # Crop second pass image data by depth
        second_pass_cropped = second_pass_image_data[depth_idx_start:depth_idx_end, :]
        modified_sv_data = second_pass_cropped.copy()
        
        # Create a mask from Pass 1 contours, adjusted for the new ping range
        pass1_combined_mask = np.zeros_like(modified_sv_data, dtype=np.uint8)
        
        # Adjust contours for the new ping range
        adjusted_contours = []
        for contour in main_dsl_contours:
            # Convert contour coordinates to ping numbers in original range
            x_coords = contour[:, :, 0].astype(np.float32)
            ping_nums = min_ping + (x_coords / cropped_image_data_for_dsl.shape[1]) * (max_ping - min_ping)
            
            # Only include contours that overlap with second pass range
            if np.min(ping_nums) < second_pass_max_ping and np.max(ping_nums) > second_pass_min_ping:
                # Map ping numbers to new image coordinates
                new_x_coords = ((ping_nums - second_pass_min_ping) / 
                              (second_pass_max_ping - second_pass_min_ping) * modified_sv_data.shape[1])
                adjusted_contour = contour.copy()
                adjusted_contour[:, :, 0] = new_x_coords
                adjusted_contours.append(adjusted_contour.astype(np.int32))
        
        if adjusted_contours:
            cv2.drawContours(pass1_combined_mask, adjusted_contours, -1, (255), thickness=cv2.FILLED)
        
        # Dilate the Pass 1 mask
        dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, PASS_1_DILATION_KERNEL_SIZE)
        dilated_pass1_mask = cv2.dilate(pass1_combined_mask, dilation_kernel, 
                                      iterations=PASS_1_DILATION_ITERATIONS)
        
        # Fill the dilated areas in the Sv data
        modified_sv_data[dilated_pass1_mask == 255] = MASK_FILL_VALUE

        if ENABLE_WAVELET_PREPROCESS and WAVELET_APPLY_TO_SECOND_PASS:
            print("Applying wavelet denoising before Pass 2 detection...")
            modified_sv_data = wavelet_denoise_sv(
                modified_sv_data,
                wavelet_name=WAVELET_NAME,
                levels=WAVELET_LEVELS,
                threshold_mode=WAVELET_THRESHOLD_MODE,
                threshold_scale=WAVELET_THRESHOLD_SCALE,
                clip_min=WAVELET_CLIP_MIN,
                clip_max=WAVELET_CLIP_MAX,
            )
        
        # Create debug directory for Pass 2
        dsl_debug_dir_pass2 = os.path.join(FIGURES_DIR, "dsl_debug_pass2")
        if not os.path.exists(dsl_debug_dir_pass2):
            os.makedirs(dsl_debug_dir_pass2)
        
        # Save debug image of modified Sv data
        save_debug_image(modified_sv_data, 'Modified Sv Data for Pass 2',
                        'pass2_0_modified_sv_data.png', dsl_debug_dir_pass2,
                        cmap='viridis', vmin=DVM_VMIN, vmax=DVM_VMAX)
        
        # Detect diffuse layers
        diffuse_dsl_contours_original, _ = detect_dsl_contours(
            None,
            modified_sv_data,
            sv_threshold_min=DIFFUSE_DSL_SV_THRESHOLD_MIN,
            sv_threshold_max=DIFFUSE_DSL_SV_THRESHOLD_MAX,
            min_contour_area=DIFFUSE_DSL_MIN_CONTOUR_AREA,
            morph_kernel_size=DIFFUSE_DSL_MORPH_KERNEL_SIZE,
            morph_close_iterations=DIFFUSE_DSL_MORPH_CLOSE_ITERATIONS,
            morph_open_iterations=DIFFUSE_DSL_MORPH_OPEN_ITERATIONS,
            contour_epsilon_factor=DIFFUSE_DSL_CONTOUR_EPSILON_FACTOR,
            debug_dvm_vmin=DVM_VMIN,
            debug_dvm_vmax=DVM_VMAX,
            figures_dir_for_debug=dsl_debug_dir_pass2,
            stage_prefix="pass2_"
        )
        print(f"Pass 2: Found {len(diffuse_dsl_contours_original)} diffuse DSL contours.")
        
        # Store original contours for CSV export (before coordinate transformation)
        diffuse_dsl_contours = diffuse_dsl_contours_original
        
        # Plot diffuse layers
        if diffuse_dsl_contours:
            diffuse_dsl_path = 'echogram_with_diffuse_dsl.png'
            plot_echogram_with_dsl(
                image_data=modified_sv_data,
                depth_start=depth_start,
                depth_stop=depth_stop,
                min_ping=second_pass_min_ping,
                max_ping=second_pass_max_ping,
                contours=diffuse_dsl_contours,
                vmin=DVM_VMIN,
                vmax=DVM_VMAX,
                title=f'{DATASET_NAME} Diffuse DVM Layers ({second_pass_min_ping}-{second_pass_max_ping})',
                save_path=diffuse_dsl_path,
                figures_dir=FIGURES_DIR,
                show_outline=show_outline,
                contour_info=None,
                x_axis_values=diffuse_plot_x_values,
                x_axis_is_time=USE_TIME_X_AXIS,
                x_axis_label='Time (UTC)' if USE_TIME_X_AXIS else 'Ping Number',
                x_axis_mode=TIME_AXIS_MODE,
            )
            print(f"Saved diffuse DSL layer visualization")
            
            # Map diffuse contours back to original ping range for combined visualization
            diffuse_dsl_contours = map_contours_between_ping_ranges(
                contours=diffuse_dsl_contours,
                source_min_ping=second_pass_min_ping,
                source_max_ping=second_pass_max_ping,
                source_image_width=modified_sv_data.shape[1],
                target_min_ping=min_ping,
                target_max_ping=max_ping,
                target_image_width=cropped_image_data_for_dsl.shape[1],
            )
            all_detected_contours.extend(diffuse_dsl_contours)

    if LOAD_REVIEWED_CONTOURS:
        expected_review_metadata = build_reviewed_contours_metadata(
            dataset_name=DATASET_NAME,
            scope=None,
            pass1_image_shape=cropped_image_data_for_dsl.shape,
            min_ping=min_ping,
            max_ping=max_ping,
            depth_start=depth_start,
            depth_stop=depth_stop,
            second_pass_min_ping=second_pass_min_ping if ENABLE_SECOND_PASS else None,
            second_pass_max_ping=second_pass_max_ping if ENABLE_SECOND_PASS else None,
            pass2_image_shape=modified_sv_data.shape if modified_sv_data is not None else None,
        )
        loaded_review_contours = load_reviewed_contours_artifact(reviewed_contours_artifact_path)
        if loaded_review_contours is not None:
            metadata_valid, metadata_reasons = validate_reviewed_contours_metadata(
                loaded_review_contours.get("metadata", {}),
                expected_review_metadata,
            )
            if metadata_valid:
                loaded_main_contours = loaded_review_contours.get("main_contours", [])
                loaded_diffuse_contours = loaded_review_contours.get("diffuse_contours", [])
                loaded_diffuse_contours_original = loaded_review_contours.get(
                    "diffuse_contours_original",
                    [],
                )

                main_dsl_contours = [contour.copy() for contour in loaded_main_contours]
                loaded_scope = loaded_review_contours.get("metadata", {}).get("scope")
                if isinstance(loaded_scope, str) and loaded_scope in {"main", "all"}:
                    layer_review_scope = loaded_scope

                if layer_review_scope == "all":
                    diffuse_dsl_contours = []
                    diffuse_dsl_contours_original = []
                elif ENABLE_SECOND_PASS:
                    if loaded_diffuse_contours:
                        diffuse_dsl_contours = [contour.copy() for contour in loaded_diffuse_contours]
                    if loaded_diffuse_contours_original:
                        diffuse_dsl_contours_original = [
                            contour.copy() for contour in loaded_diffuse_contours_original
                        ]

                reviewed_contours_loaded = True
                layer_review_applied = True
                reviewed_contours_load_metadata = loaded_review_contours.get("metadata", {})
                if ENABLE_LAYER_REVIEW:
                    review_output_figures_dir = reviewed_contours_artifact_dir
                    os.makedirs(review_output_figures_dir, exist_ok=True)
                reviewed_contours_status["loaded_from_artifact"] = True
                reviewed_contours_status["load_status"] = "loaded"
                reviewed_contours_status["load_reason"] = "loaded_valid_artifact"
                print(
                    f"Loaded reviewed contours from artifact: {reviewed_contours_artifact_path}"
                )
                print(
                    "Reviewed contour load status: LOADED "
                    f"(dataset={DATASET_NAME}, artifact={reviewed_contours_artifact_path})"
                )
                if ENABLE_LAYER_REVIEW and SKIP_LAYER_REVIEW_IF_LOADED:
                    print("Skipping interactive review because loaded contours are being reused.")
            else:
                reviewed_contours_status["load_status"] = "refused"
                reviewed_contours_status["load_reason"] = "metadata_mismatch"
                reviewed_contours_status["metadata_mismatch_reasons"] = list(metadata_reasons)
                print(
                    "Reviewed contour artifact metadata mismatch; running normal detection/review "
                    f"instead ({', '.join(metadata_reasons)})."
                )
                print(
                    "Reviewed contour load status: REFUSED "
                    f"(dataset={DATASET_NAME}, reason=metadata_mismatch)"
                )
        else:
            reviewed_contours_status["load_status"] = "refused"
            reviewed_contours_status["load_reason"] = "artifact_missing_or_unreadable"
            print(
                "Reviewed contour load status: REFUSED "
                f"(dataset={DATASET_NAME}, reason=artifact_missing_or_unreadable)"
            )

    # 3. Optional interactive review (main/all), before final outputs.
    if ENABLE_LAYER_REVIEW and not (reviewed_contours_loaded and SKIP_LAYER_REVIEW_IF_LOADED):
        layer_review_scope = resolve_layer_review_scope(LAYER_REVIEW_SCOPE)
        print(f"\nLayer review enabled (scope: {layer_review_scope})")

        main_review_result = None
        diffuse_review_result = None

        if layer_review_scope == "main" and main_dsl_contours:
            main_review_result = review_contours_interactively(
                image_data=cropped_image_data_for_dsl,
                contours=main_dsl_contours,
                depth_start=depth_start,
                depth_stop=depth_stop,
                min_ping=min_ping,
                max_ping=max_ping,
                title=f"{DATASET_NAME} Main Layer Review",
                x_axis_values=main_plot_x_values,
                x_axis_is_time=USE_TIME_X_AXIS,
                x_axis_label='Time (UTC)' if USE_TIME_X_AXIS else 'Ping Number',
                x_axis_mode=TIME_AXIS_MODE,
                layer_prefix="Main",
                background_vmin=DVM_VMIN,
                background_vmax=DVM_VMAX,
            )
            if main_review_result["accepted"]:
                main_dsl_contours = main_review_result["contours"]
                layer_review_applied = True
                print(
                    f"Applied main review: {main_review_result['initial_count']} -> "
                    f"{main_review_result['final_count']} contours."
                )

        if layer_review_scope == "all":
            all_scope_contours = [contour.copy() for contour in main_dsl_contours]
            if ENABLE_SECOND_PASS and diffuse_dsl_contours:
                all_scope_contours.extend([contour.copy() for contour in diffuse_dsl_contours])

            main_review_result = review_contours_interactively(
                image_data=cropped_image_data_for_dsl,
                contours=all_scope_contours,
                depth_start=depth_start,
                depth_stop=depth_stop,
                min_ping=min_ping,
                max_ping=max_ping,
                title=f"{DATASET_NAME} All Layer Review",
                x_axis_values=main_plot_x_values,
                x_axis_is_time=USE_TIME_X_AXIS,
                x_axis_label='Time (UTC)' if USE_TIME_X_AXIS else 'Ping Number',
                x_axis_mode=TIME_AXIS_MODE,
                layer_prefix="All",
                background_vmin=DVM_VMIN,
                background_vmax=DVM_VMAX,
            )
            if main_review_result["accepted"]:
                main_dsl_contours = main_review_result["contours"]
                diffuse_dsl_contours = []
                diffuse_dsl_contours_original = []
                layer_review_applied = True
                print(
                    f"Applied all-layer review: {main_review_result['initial_count']} -> "
                    f"{main_review_result['final_count']} contours."
                )

        if layer_review_applied:
            review_output_figures_dir = os.path.join(FIGURES_DIR, layer_review_output_subdir)
            os.makedirs(review_output_figures_dir, exist_ok=True)
            review_manifest = {
                "dataset_name": DATASET_NAME,
                "scope": layer_review_scope,
                "main": summarize_review_result_for_manifest(main_review_result),
                "diffuse": summarize_review_result_for_manifest(diffuse_review_result),
                "review_output_subdir_config": LAYER_REVIEW_OUTPUT_SUBDIR,
                "review_output_subdir_resolved": layer_review_output_subdir,
            }
            save_layer_review_manifest(review_output_figures_dir, review_manifest)

            # Save reviewed versions of main/diffuse overlays immediately.
            if main_dsl_contours:
                reviewed_main_title = (
                    f'{DATASET_NAME} All Reviewed Layers ({min_ping}-{max_ping})'
                    if layer_review_scope == "all"
                    else f'{DATASET_NAME} Main DVM Layers ({min_ping}-{max_ping}) [Reviewed]'
                )
                plot_echogram_with_dsl(
                    image_data=cropped_image_data_for_dsl,
                    depth_start=depth_start,
                    depth_stop=depth_stop,
                    min_ping=min_ping,
                    max_ping=max_ping,
                    contours=main_dsl_contours,
                    vmin=DVM_VMIN,
                    vmax=DVM_VMAX,
                    title=reviewed_main_title,
                    save_path='echogram_with_main_dsl.png',
                    figures_dir=review_output_figures_dir,
                    show_outline=show_outline,
                    contour_info=None,
                    x_axis_values=main_plot_x_values,
                    x_axis_is_time=USE_TIME_X_AXIS,
                    x_axis_label='Time (UTC)' if USE_TIME_X_AXIS else 'Ping Number',
                    x_axis_mode=TIME_AXIS_MODE,
                )
                print("Saved reviewed main DSL visualization")

            if ENABLE_SECOND_PASS and diffuse_dsl_contours_original and modified_sv_data is not None:
                plot_echogram_with_dsl(
                    image_data=modified_sv_data,
                    depth_start=depth_start,
                    depth_stop=depth_stop,
                    min_ping=second_pass_min_ping,
                    max_ping=second_pass_max_ping,
                    contours=diffuse_dsl_contours_original,
                    vmin=DVM_VMIN,
                    vmax=DVM_VMAX,
                    title=(
                        f'{DATASET_NAME} Diffuse DVM Layers '
                        f'({second_pass_min_ping}-{second_pass_max_ping}) [Reviewed]'
                    ),
                    save_path='echogram_with_diffuse_dsl.png',
                    figures_dir=review_output_figures_dir,
                    show_outline=show_outline,
                    contour_info=None,
                    x_axis_values=diffuse_plot_x_values,
                    x_axis_is_time=USE_TIME_X_AXIS,
                    x_axis_label='Time (UTC)' if USE_TIME_X_AXIS else 'Ping Number',
                    x_axis_mode=TIME_AXIS_MODE,
                )
                print("Saved reviewed diffuse DSL visualization")

    if layer_review_applied and SAVE_REVIEWED_CONTOURS:
        artifact_scope = layer_review_scope
        if artifact_scope is None and reviewed_contours_load_metadata is not None:
            artifact_scope = reviewed_contours_load_metadata.get("scope")

        contour_artifact_metadata = build_reviewed_contours_metadata(
            dataset_name=DATASET_NAME,
            scope=artifact_scope,
            pass1_image_shape=cropped_image_data_for_dsl.shape,
            min_ping=min_ping,
            max_ping=max_ping,
            depth_start=depth_start,
            depth_stop=depth_stop,
            second_pass_min_ping=second_pass_min_ping if ENABLE_SECOND_PASS else None,
            second_pass_max_ping=second_pass_max_ping if ENABLE_SECOND_PASS else None,
            pass2_image_shape=modified_sv_data.shape if modified_sv_data is not None else None,
        )
        reviewed_contours_artifact_saved_path = save_reviewed_contours_artifact(
            artifact_path=reviewed_contours_artifact_path,
            metadata=contour_artifact_metadata,
            main_contours=main_dsl_contours,
            diffuse_contours=diffuse_dsl_contours if ENABLE_SECOND_PASS else [],
            diffuse_contours_original=diffuse_dsl_contours_original if ENABLE_SECOND_PASS else [],
        )
    elif SAVE_REVIEWED_CONTOURS and not layer_review_applied:
        print(
            "Skipping reviewed contour artifact save because no reviewed contours were applied."
        )

    # Always refresh reviewed overlays after review/loaded contours are applied.
    # This prevents stale first-pass overlays from being mistaken as reviewed output.
    if layer_review_applied:
        if main_dsl_contours:
            reviewed_main_title = (
                f'{DATASET_NAME} All Reviewed Layers ({min_ping}-{max_ping})'
                if layer_review_scope == "all"
                else f'{DATASET_NAME} Main DVM Layers ({min_ping}-{max_ping}) [Reviewed]'
            )
            plot_echogram_with_dsl(
                image_data=cropped_image_data_for_dsl,
                depth_start=depth_start,
                depth_stop=depth_stop,
                min_ping=min_ping,
                max_ping=max_ping,
                contours=main_dsl_contours,
                vmin=DVM_VMIN,
                vmax=DVM_VMAX,
                title=reviewed_main_title,
                save_path='echogram_with_main_dsl.png',
                figures_dir=review_output_figures_dir,
                show_outline=show_outline,
                contour_info=None,
                x_axis_values=main_plot_x_values,
                x_axis_is_time=USE_TIME_X_AXIS,
                x_axis_label='Time (UTC)' if USE_TIME_X_AXIS else 'Ping Number',
                x_axis_mode=TIME_AXIS_MODE,
            )
            print("Saved reviewed main DSL visualization")

        if (
            ENABLE_SECOND_PASS
            and diffuse_dsl_contours_original
            and modified_sv_data is not None
        ):
            plot_echogram_with_dsl(
                image_data=modified_sv_data,
                depth_start=depth_start,
                depth_stop=depth_stop,
                min_ping=second_pass_min_ping,
                max_ping=second_pass_max_ping,
                contours=diffuse_dsl_contours_original,
                vmin=DVM_VMIN,
                vmax=DVM_VMAX,
                title=(
                    f'{DATASET_NAME} Diffuse DVM Layers '
                    f'({second_pass_min_ping}-{second_pass_max_ping}) [Reviewed]'
                ),
                save_path='echogram_with_diffuse_dsl.png',
                figures_dir=review_output_figures_dir,
                show_outline=show_outline,
                contour_info=None,
                x_axis_values=diffuse_plot_x_values,
                x_axis_is_time=USE_TIME_X_AXIS,
                x_axis_label='Time (UTC)' if USE_TIME_X_AXIS else 'Ping Number',
                x_axis_mode=TIME_AXIS_MODE,
            )
            print("Saved reviewed diffuse DSL visualization")

    # Refresh the combined contour list after optional review edits.
    all_detected_contours = list(main_dsl_contours)
    if ENABLE_SECOND_PASS and diffuse_dsl_contours:
        all_detected_contours.extend(diffuse_dsl_contours)

    results_figures_dir = review_output_figures_dir if layer_review_applied else FIGURES_DIR

    layer_speed_metrics = []

    # 4. Plot combined visualization with all layers, maintaining distinct colors
    if main_dsl_contours or (ENABLE_SECOND_PASS and diffuse_dsl_contours):
        # Create a composite image for the combined visualization
        # Use original data for the entire ping range to maintain consistent background
        
        # Determine the full ping range that covers both passes
        full_min_ping = min(min_ping, second_pass_min_ping)
        full_max_ping = max(max_ping, second_pass_max_ping)
        
        # Get the full echogram data for the combined ping range
        full_ping_mask = (echogram['Ping_index'] >= full_min_ping) & (echogram['Ping_index'] <= full_max_ping)
        full_echogram = echogram[full_ping_mask]
        _, full_image_data = prepare_sample_data(full_echogram)
        combined_plot_x_values = create_plot_time_axis_values(full_echogram) if USE_TIME_X_AXIS else None
        
        # Crop by depth
        full_cropped_image = full_image_data[depth_idx_start:depth_idx_end, :]
        
        # Get distinct colors for main layers
        main_colors = get_distinct_colors(len(main_dsl_contours))
        
        # Get distinct colors for diffuse layers if they exist
        diffuse_colors = []
        if ENABLE_SECOND_PASS and diffuse_dsl_contours:
            diffuse_colors = get_distinct_colors(len(diffuse_dsl_contours))

        # Adjust contours and prepare them for combined plotting
        adjusted_all_contours = []

        # Adjust main contours to the full image space
        adjusted_main_contours = []
        for contour in main_dsl_contours:
            x_coords = contour[:, :, 0].astype(np.float32)
            # Convert from the first pass image coordinates to ping numbers
            ping_nums = min_ping + (x_coords / cropped_image_data_for_dsl.shape[1]) * (max_ping - min_ping)
            # Convert ping numbers to the full combined image coordinates
            new_x_coords = ((ping_nums - full_min_ping) / (full_max_ping - full_min_ping)) * full_cropped_image.shape[1]
            adjusted_contour = contour.copy()
            adjusted_contour[:, :, 0] = new_x_coords
            adjusted_main_contours.append(adjusted_contour.astype(np.int32))

        if adjusted_main_contours:
            adjusted_all_contours.extend(adjusted_main_contours)

        # Adjust diffuse contours to the full image space
        if ENABLE_SECOND_PASS and diffuse_dsl_contours:
            adjusted_diffuse_contours_final = []
            # Note: diffuse_dsl_contours have already been mapped to the first pass image space
            for contour in diffuse_dsl_contours:
                x_coords = contour[:, :, 0].astype(np.float32)
                # Convert from the first pass image coordinates to ping numbers
                ping_nums = min_ping + (x_coords / cropped_image_data_for_dsl.shape[1]) * (max_ping - min_ping)
                # Convert ping numbers to the full combined image coordinates
                new_x_coords = ((ping_nums - full_min_ping) / (full_max_ping - full_min_ping)) * full_cropped_image.shape[1]
                adjusted_contour = contour.copy()
                adjusted_contour[:, :, 0] = new_x_coords
                adjusted_diffuse_contours_final.append(adjusted_contour.astype(np.int32))

            if adjusted_diffuse_contours_final:
                adjusted_all_contours.extend(adjusted_diffuse_contours_final)
        
        # Build title for the combined plot
        if ENABLE_SECOND_PASS and diffuse_dsl_contours:
            combined_title = f'{DATASET_NAME} Combined DVM Layers (Main: {min_ping}-{max_ping}, Diffuse: {second_pass_min_ping}-{second_pass_max_ping})'
        else:
            combined_title = f'{DATASET_NAME} All Detected DVM Layers ({full_min_ping}-{full_max_ping})'

        # Separate the adjusted contours back into main and diffuse groups for plotting
        num_main_contours = len(main_dsl_contours)
        adjusted_main_dsl_contours = adjusted_all_contours[:num_main_contours]
        adjusted_diffuse_dsl_contours = adjusted_all_contours[num_main_contours:]
        
        final_contour_info = []
        if adjusted_main_dsl_contours:
            main_colors = get_distinct_colors(len(adjusted_main_dsl_contours))
            final_contour_info.append((adjusted_main_dsl_contours, main_colors, 'Main'))
        
        if adjusted_diffuse_dsl_contours:
            diffuse_colors = get_distinct_colors(len(adjusted_diffuse_dsl_contours))
            final_contour_info.append((adjusted_diffuse_dsl_contours, diffuse_colors, 'Diffuse'))

        combined_dsl_path = 'echogram_with_all_dsl.png'
        plot_echogram_with_dsl(
            image_data=full_cropped_image,  # Use the full original data
            depth_start=depth_start,
            depth_stop=depth_stop,
            min_ping=full_min_ping,  # Use the full ping range
            max_ping=full_max_ping,
            contours=adjusted_all_contours,  # Pass all contours, though ignored by plotting logic
            vmin=DVM_VMIN,
            vmax=DVM_VMAX,
            title=combined_title,
            save_path=combined_dsl_path,
            figures_dir=results_figures_dir,
            show_outline=show_outline,
            contour_info=final_contour_info,
            x_axis_values=combined_plot_x_values,
            x_axis_is_time=USE_TIME_X_AXIS,
            x_axis_label='Time (UTC)' if USE_TIME_X_AXIS else 'Ping Number',
            x_axis_mode=TIME_AXIS_MODE,
        )
        print(f"Saved combined DSL layer visualization")

        expected_direction = infer_expected_direction(DATASET_NAME)
        direction_name = direction_display_name(expected_direction)
        direction_multiplier = expected_direction_multiplier(expected_direction)

        # Calculate vertical movement and speed for each displayed layer.
        full_ping_values, full_elapsed_minutes, full_timestamps = create_ping_time_arrays(full_echogram)
        time_origin_utc = full_timestamps.iloc[0] if len(full_timestamps) > 0 else None
        for contour_list, color_list, label_prefix in final_contour_info:
            for idx, (contour, color) in enumerate(zip(contour_list, color_list), start=1):
                layer_metrics = calculate_layer_vertical_metrics(
                    contour=contour,
                    image_shape=full_cropped_image.shape,
                    min_ping=full_min_ping,
                    max_ping=full_max_ping,
                    depth_start=depth_start,
                    depth_stop=depth_stop,
                    ping_values=full_ping_values,
                    elapsed_minutes=full_elapsed_minutes,
                    time_origin_utc=time_origin_utc,
                )
                if layer_metrics is None:
                    continue
                layer_metrics['label'] = f'{label_prefix} Layer {idx}'
                layer_metrics['layer_type'] = label_prefix.lower()
                layer_metrics['color'] = color
                if layer_metrics['layer_type'] == 'main':
                    layer_metrics['_contour_index'] = idx - 1
                layer_metrics['cast_direction'] = expected_direction
                add_expected_direction_speed_fields(
                    layer_metrics=layer_metrics,
                    direction_multiplier=direction_multiplier,
                )
                layer_speed_metrics.append(layer_metrics)

        if layer_speed_metrics:
            exportable_layer_metrics = [
                {
                    key: value
                    for key, value in metric.items()
                    if not str(key).startswith('_')
                }
                for metric in layer_speed_metrics
            ]
            layer_metrics_df = pd.DataFrame(exportable_layer_metrics)
            layer_metrics_csv = os.path.join(results_figures_dir, 'dsl_layer_speed_vertical_summary.csv')
            layer_metrics_df.to_csv(layer_metrics_csv, index=False)
            print(f"Saved layer speed/vertical summary to: {layer_metrics_csv}")

            comparison_columns = [
                'label',
                'layer_type',
                'cast_direction',
                'primary_speed_metric',
                'speed_m_per_min_display',
                'speed_m_per_min_regression_display',
                'speed_m_per_min_windowed_display',
                'contour_start_time_utc',
                'contour_end_time_utc',
                'motion_start_time_utc',
                'motion_end_time_utc',
                'contour_duration_min',
                'motion_duration_min',
                'motion_detected',
                'motion_confidence',
                'opposes_expected_direction',
                'opposes_expected_direction_regression',
                'opposes_expected_direction_windowed',
                'primary_opposes_expected_direction',
            ]
            comparison_columns = [col for col in comparison_columns if col in layer_metrics_df.columns]
            if comparison_columns:
                method_comparison_csv = os.path.join(
                    results_figures_dir,
                    'dsl_layer_speed_method_comparison.csv',
                )
                layer_metrics_df.loc[:, comparison_columns].to_csv(method_comparison_csv, index=False)
                print(f"Saved layer speed method comparison to: {method_comparison_csv}")

            combined_image_path = os.path.join(results_figures_dir, combined_dsl_path)
            speed_figure_path = os.path.join(results_figures_dir, 'echogram_and_layer_speed.png')
            save_layer_speed_figure(
                echogram_image_path=combined_image_path,
                layer_metrics=layer_speed_metrics,
                output_path=speed_figure_path,
                dataset_name=DATASET_NAME,
                expected_direction=expected_direction,
                plot_title=f'All Layers: {direction_name} Trend Speed (m/min)',
            )

            main_layer_metrics = [
                dict(metric) for metric in layer_speed_metrics
                if metric['layer_type'] == 'main'
            ]
            if main_layer_metrics:
                main_image_path = os.path.join(results_figures_dir, 'echogram_with_main_dsl.png')
                main_speed_figure_path = os.path.join(results_figures_dir, 'echogram_and_main_layer_speed.png')
                save_layer_speed_figure(
                    echogram_image_path=main_image_path,
                    layer_metrics=main_layer_metrics,
                    output_path=main_speed_figure_path,
                    dataset_name=DATASET_NAME,
                    expected_direction=expected_direction,
                    plot_title=f'Main Layers: {direction_name} Trend Speed (m/min)',
                )

                ranked_main_metrics, ranked_main_contours, ranked_main_colors = (
                    build_time_ranked_main_layer_outputs(main_layer_metrics, main_dsl_contours)
                )
                if ranked_main_metrics and ranked_main_contours:
                    ranked_main_overlay_filename = 'echogram_with_main_dsl_time_ranked.png'
                    plot_echogram_with_dsl(
                        image_data=cropped_image_data_for_dsl,
                        depth_start=depth_start,
                        depth_stop=depth_stop,
                        min_ping=min_ping,
                        max_ping=max_ping,
                        contours=ranked_main_contours,
                        vmin=DVM_VMIN,
                        vmax=DVM_VMAX,
                        title=f'{DATASET_NAME} Main DVM Layers ({min_ping}-{max_ping}) [Time-Ranked]',
                        save_path=ranked_main_overlay_filename,
                        figures_dir=results_figures_dir,
                        show_outline=show_outline,
                        contour_info=[(ranked_main_contours, ranked_main_colors, 'Main')],
                        x_axis_values=main_plot_x_values,
                        x_axis_is_time=USE_TIME_X_AXIS,
                        x_axis_label='Time (UTC)' if USE_TIME_X_AXIS else 'Ping Number',
                        x_axis_mode=TIME_AXIS_MODE,
                    )
                    print("Saved time-ranked main DSL visualization")

                    ranked_main_speed_figure_path = os.path.join(
                        results_figures_dir,
                        'echogram_and_main_layer_speed_time_ranked.png',
                    )
                    save_layer_speed_figure(
                        echogram_image_path=os.path.join(
                            results_figures_dir,
                            ranked_main_overlay_filename,
                        ),
                        layer_metrics=ranked_main_metrics,
                        output_path=ranked_main_speed_figure_path,
                        dataset_name=DATASET_NAME,
                        expected_direction=expected_direction,
                        plot_title=f'Main Layers (time-ranked): {direction_name} Trend Speed (m/min)',
                    )

            diffuse_layer_metrics = [
                metric for metric in layer_speed_metrics
                if metric['layer_type'] == 'diffuse'
            ]
            if diffuse_layer_metrics:
                diffuse_image_path = os.path.join(results_figures_dir, 'echogram_with_diffuse_dsl.png')
                diffuse_speed_figure_path = os.path.join(results_figures_dir, 'echogram_and_diffuse_layer_speed.png')
                save_layer_speed_figure(
                    echogram_image_path=diffuse_image_path,
                    layer_metrics=diffuse_layer_metrics,
                    output_path=diffuse_speed_figure_path,
                    dataset_name=DATASET_NAME,
                    expected_direction=expected_direction,
                    plot_title=f'Diffuse Layers: {direction_name} Trend Speed (m/min)',
                )
    
    # Export main DSL layers to CSV files (skip diffuse layers for now)
    if EXPORT_BOOLEAN_CSV and main_dsl_contours:
         # Skip Sv CSV export for now (takes too long)
         #print("\nExporting main DSL layers to CSV files...")
         #main_csv_dir = os.path.join(FIGURES_DIR, f'{DATASET_NAME}_main_dsl_layers_sv_csv')
         #export_dsl_layers_to_csv(
         #   contours=main_dsl_contours,
         #   original_echogram_df=echogram,
         #   min_ping=min_ping,
         #   max_ping=max_ping,
         #   depth_start=depth_start,
         #   depth_stop=depth_stop,
         #   image_shape=cropped_image_data_for_dsl.shape,
         #   output_dir=main_csv_dir,
         #   layer_prefix="main_dsl"
         #)
        
        # Export boolean CSV files (much faster)
        print("\nExporting main DSL layers to boolean CSV files...")
        main_boolean_csv_dir = os.path.join(results_figures_dir, f'{DATASET_NAME}_main_dsl_layers_boolean_csv')
        export_dsl_layers_to_boolean_csv(
            contours=main_dsl_contours,
            original_echogram_df=echogram,
            min_ping=min_ping,
            max_ping=max_ping,
            depth_start=depth_start,
            depth_stop=depth_stop,
            image_shape=cropped_image_data_for_dsl.shape,
            output_dir=main_boolean_csv_dir,
            layer_prefix="main_dsl"
            )
    
    # Export diffuse DSL layers to boolean CSV files if they exist
    if EXPORT_BOOLEAN_CSV and ENABLE_SECOND_PASS and diffuse_dsl_contours_original:
        print("\nExporting diffuse DSL layers to boolean CSV files...")
        
        # Get the second pass echogram data for export
        second_pass_ping_mask = (echogram['Ping_index'] >= second_pass_min_ping) & \
                              (echogram['Ping_index'] <= second_pass_max_ping)
        second_pass_echogram_for_export = echogram[second_pass_ping_mask]
        
        # Use the original diffuse contours (in modified_sv_data coordinate space)
        diffuse_boolean_csv_dir = os.path.join(results_figures_dir, f'{DATASET_NAME}_diffuse_dsl_layers_boolean_csv')
        export_dsl_layers_to_boolean_csv(
            contours=diffuse_dsl_contours_original,
            original_echogram_df=second_pass_echogram_for_export,
            min_ping=second_pass_min_ping,
            max_ping=second_pass_max_ping,
            depth_start=depth_start,
            depth_stop=depth_stop,
            image_shape=modified_sv_data.shape,  # Use the second pass image shape
            output_dir=diffuse_boolean_csv_dir,
            layer_prefix="diffuse_dsl"
        )
    elif not EXPORT_BOOLEAN_CSV:
        print("\nSkipping boolean CSV export (EXPORT_BOOLEAN_CSV=False).")

    if not reviewed_contours_status.get("requested_load", False) and layer_review_applied:
        reviewed_contours_status["load_reason"] = "manual_review_applied"

    reviewed_contours_status["reviewed_contours_loaded"] = bool(reviewed_contours_loaded)
    reviewed_contours_status["layer_review_applied"] = bool(layer_review_applied)
    reviewed_contours_status["layer_review_scope"] = layer_review_scope
    reviewed_contours_status["results_figures_dir"] = results_figures_dir
    reviewed_contours_status_path = save_reviewed_contours_status(
        FIGURES_DIR,
        reviewed_contours_status,
    )

    copied_artifacts_to_output = copy_result_artifacts_to_output_folder(
        source_dir=results_figures_dir,
        output_dir=FIGURES_DIR,
        artifact_filenames=[
            "echogram_with_main_dsl.png",
            "echogram_with_main_dsl_time_ranked.png",
            "echogram_with_diffuse_dsl.png",
            "echogram_with_all_dsl.png",
            "echogram_and_layer_speed.png",
            "echogram_and_main_layer_speed.png",
            "echogram_and_main_layer_speed_time_ranked.png",
            "echogram_and_diffuse_layer_speed.png",
            "dsl_layer_speed_vertical_summary.csv",
            "dsl_layer_speed_method_comparison.csv",
        ],
    )
    
    return {
        'ping_time_map': ping_time_map,
        'layer_speed_metrics': layer_speed_metrics,
        'main_dsl_contours': main_dsl_contours,
        'diffuse_dsl_contours': diffuse_dsl_contours if ENABLE_SECOND_PASS else [],
        'diffuse_dsl_contours_original': diffuse_dsl_contours_original if ENABLE_SECOND_PASS else [],
        'all_contours': all_detected_contours,
        'depth_range': (depth_start, depth_stop),
        'ping_range': (min_ping, max_ping),
        'second_pass_ping_range': (second_pass_min_ping, second_pass_max_ping) if ENABLE_SECOND_PASS else None,
        'layer_review_applied': layer_review_applied,
        'layer_review_scope': layer_review_scope,
        'results_figures_dir': results_figures_dir,
        'layer_review_manifest': review_manifest,
        'reviewed_contours_loaded': reviewed_contours_loaded,
        'reviewed_contours_status': reviewed_contours_status,
        'reviewed_contours_status_path': reviewed_contours_status_path,
        'copied_artifacts_to_output': copied_artifacts_to_output,
        'reviewed_contours_artifact_path': (
            reviewed_contours_artifact_saved_path
            if reviewed_contours_artifact_saved_path is not None
            else (reviewed_contours_artifact_path if reviewed_contours_loaded else None)
        ),
    }

if __name__ == "__main__":
    results = process_echogram_with_dsl_detection(
        start_ping=START_PING,
        end_ping=END_PING,
        show_outline=False # Set to True to show the contour outline
    )
    
    # CSV export is handled inside process_echogram_with_dsl_detection
    # Main DSL layers are exported as separate .sv.csv files for Echoview import