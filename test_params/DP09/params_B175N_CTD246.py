FILE_PATH = "test_csvs/DP09/B175N_CTD246_18kHz.sv.csv"

# Visualization parameters (Default Echogram)
dmin = -75  # Display minimum value from Echoview
dmax = None # Display maximum where dmax = dmin + drange
drange = 10 # Display range from Echoview
USE_TIME_X_AXIS = True  # Default: show UTC time on plot x-axes (instead of ping index)
TIME_AXIS_MODE = "linear"  # "true_time" or "linear"

# Output organization (optional)
GROUP_OUTPUTS_BY_CRUISE = True  # Save outputs to <root>/<CRUISE>/<DATASET> when cruise can be resolved
CRUISE_NAME = None  # Optional manual override, e.g. "DP09"

# Optional interactive layer review (post-processing)
ENABLE_LAYER_REVIEW = False
LAYER_REVIEW_SCOPE = "choose_each_run"  # "main", "all", or "choose_each_run"
LAYER_REVIEW_OUTPUT_SUBDIR = "reviewed"
SAVE_REVIEWED_CONTOURS = True
LOAD_REVIEWED_CONTOURS = True
SKIP_LAYER_REVIEW_IF_LOADED = True
REVIEWED_CONTOURS_FILENAME = "reviewed_contours.npz"
REVIEWED_CONTOURS_SUBDIR = None

# Optional wavelet preprocessing (disabled by default)
ENABLE_WAVELET_PREPROCESS = False
WAVELET_NAME = "db4"
WAVELET_LEVELS = 3
WAVELET_THRESHOLD_MODE = "soft"  # "soft" or "hard"
WAVELET_THRESHOLD_SCALE = 1.0
WAVELET_CLIP_MIN = -90.0
WAVELET_CLIP_MAX = -30.0
WAVELET_APPLY_TO_SECOND_PASS = False

# Analysis Ping Range (Limit our x-axis range)
START_PING = 1905    # Starting ping number for analysis
END_PING = 2500      # Ending ping number for analysis (showing 200 pings)

# Optional time-based range override (UTC). If set, these override START_PING/END_PING.
# Accepts "HH:MM[:SS]" on dataset date, or full datetime (e.g., "2022-08-03 11:20:00+00:00")
START_TIME_UTC = None
END_TIME_UTC = None

# DSL/DVM Analysis Depth Range (Limit our y-axis range)
DSL_DEPTH_START = 150  # Starting depth in meters for DSL/DVM analysis
DSL_DEPTH_STOP = 650 #430   # Stopping depth in meters for DSL/DVM analysis

# DVM Detection Parameters (Used for the other echograms)
DVM_VMIN = -75
DVM_VMAX = -65

# DSL Detection Parameters 
DSL_SV_THRESHOLD_MIN = -71.8  #-75.5 # Minimum Sv value for a pixel to be considered part of a DSL
DSL_SV_THRESHOLD_MAX = -60.9  #-65.5 # Maximum Sv value for a pixel to be considered part of a DSL

# OpenCV DSL Post-processing
DSL_MIN_CONTOUR_AREA = 10000#375  #10000 # Minimum pixel area to be considered a valid DSL contour

# Contour Smoothing Parameters
DSL_CONTOUR_EPSILON_FACTOR = 0.00005  # Factor for contour approximation (larger = smoother contours)

# Note: Keeping morphological parameters at 0 to preserve distinct layers
DSL_MORPH_KERNEL_SIZE = (3, 3)  # Only used if needed in future
DSL_MORPH_CLOSE_ITERATIONS = 1  # Disabled to preserve layer separation
DSL_MORPH_OPEN_ITERATIONS = 1   # Disabled to preserve layer separation


# Toggle CSV export to save runtime during shape tuning.
EXPORT_BOOLEAN_CSV = False
# --- Parameters for Second Pass (Diffuse Layers) ---
ENABLE_SECOND_PASS = False  # Flag to enable/disable second pass detection

# Second Pass Ping Range (Can be different from first pass)
SECOND_PASS_START_PING = None  # If None, uses START_PING
SECOND_PASS_END_PING = None    # If None, uses END_PING
# Optional time-based second-pass override (UTC). If set, these override second pass ping bounds.
SECOND_PASS_START_TIME_UTC = None
SECOND_PASS_END_TIME_UTC = None

# Value to fill masked areas from Pass 1 (should be outside Pass 2 thresholds)
MASK_FILL_VALUE = -90.0

# Dilation of Pass 1 contours before masking
PASS_1_DILATION_KERNEL_SIZE = (12, 12)  
PASS_1_DILATION_ITERATIONS = 1

# Sv Thresholds for Diffuse Layers
DIFFUSE_DSL_SV_THRESHOLD_MIN = -74.0  # More sensitive threshold
DIFFUSE_DSL_SV_THRESHOLD_MAX = -55.0  # Upper bound for diffuse layers

# OpenCV Post-processing for Diffuse Layers
DIFFUSE_DSL_MIN_CONTOUR_AREA = 2500
DIFFUSE_DSL_MORPH_KERNEL_SIZE = (3, 3)
DIFFUSE_DSL_MORPH_CLOSE_ITERATIONS = 2
DIFFUSE_DSL_MORPH_OPEN_ITERATIONS = 1
DIFFUSE_DSL_CONTOUR_EPSILON_FACTOR = 0.00005  # Different smoothing for diffuse layers