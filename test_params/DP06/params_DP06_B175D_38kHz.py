# Auto-generated per-site params seeded from batch_master_config.py.
# Tune this file independently when this site needs custom thresholds/ranges.
FILE_PATH = 'test_csvs/DP06/DP06_B175D_38kHz.sv.csv'

# Visualization parameters
dmin = -75
dmax = None
drange = 10
USE_TIME_X_AXIS = True
TIME_AXIS_MODE = "linear"  # "true_time" or "linear"

# Optional interactive layer review (post-processing)
ENABLE_LAYER_REVIEW = True
LAYER_REVIEW_SCOPE = "choose_each_run"  # "main", "all", or "choose_each_run"
LAYER_REVIEW_OUTPUT_SUBDIR = "reviewed"
SAVE_REVIEWED_CONTOURS = True
LOAD_REVIEWED_CONTOURS = True
SKIP_LAYER_REVIEW_IF_LOADED = True
REVIEWED_CONTOURS_FILENAME = "reviewed_contours.npz"
REVIEWED_CONTOURS_SUBDIR = None
GROUP_OUTPUTS_BY_CRUISE = True
CRUISE_NAME = 'DP06'

# Optional wavelet preprocessing (disabled by default)
ENABLE_WAVELET_PREPROCESS = False
WAVELET_NAME = "db4"
WAVELET_LEVELS = 3
WAVELET_THRESHOLD_MODE = "soft"  # "soft" or "hard"
WAVELET_THRESHOLD_SCALE = 1.0
WAVELET_CLIP_MIN = -90.0
WAVELET_CLIP_MAX = -30.0
WAVELET_APPLY_TO_SECOND_PASS = False

# Analysis range
START_PING = None
END_PING = None
START_TIME_UTC = None
END_TIME_UTC = None

# Depth range
DSL_DEPTH_START = 150
DSL_DEPTH_STOP = 550

# DVM visualization limits
DVM_VMIN = -75
DVM_VMAX = -65

# Main DSL detection thresholds
DSL_SV_THRESHOLD_MIN = -75.6
DSL_SV_THRESHOLD_MAX = -58.0

# Main DSL contour filtering and cleanup
DSL_MIN_CONTOUR_AREA = 1000
DSL_CONTOUR_EPSILON_FACTOR = 5e-05
DSL_MORPH_KERNEL_SIZE = (2, 2)
DSL_MORPH_CLOSE_ITERATIONS = 2
DSL_MORPH_OPEN_ITERATIONS = 1

# Optional second pass and exports
EXPORT_BOOLEAN_CSV = False
ENABLE_SECOND_PASS = False
SECOND_PASS_START_PING = None
SECOND_PASS_END_PING = None
SECOND_PASS_START_TIME_UTC = None
SECOND_PASS_END_TIME_UTC = None
MASK_FILL_VALUE = -90.0
PASS_1_DILATION_KERNEL_SIZE = (7, 7)
PASS_1_DILATION_ITERATIONS = 1

# Diffuse layer detection
DIFFUSE_DSL_SV_THRESHOLD_MIN = -74.0
DIFFUSE_DSL_SV_THRESHOLD_MAX = -55.0
DIFFUSE_DSL_MIN_CONTOUR_AREA = 2000
DIFFUSE_DSL_MORPH_KERNEL_SIZE = (3, 3)
DIFFUSE_DSL_MORPH_CLOSE_ITERATIONS = 1
DIFFUSE_DSL_MORPH_OPEN_ITERATIONS = 1
DIFFUSE_DSL_CONTOUR_EPSILON_FACTOR = 5e-05
