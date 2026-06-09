# OpenCV Layer Tracking

Python pipeline for detecting DSL/DVM layers from Echoview `.sv.csv` exports, reviewing contours interactively, and producing per-dataset and cross-dataset metrics.

All commands below assume you run from the repository root.

## Repository Scope (Current GitHub Upload)

This repository is currently a code-focused upload:

- Included: core pipeline scripts and configs.
- Excluded: most params collections, raw input data, and generated outputs/results.

In practice, you can run the code when you supply your own params files and `.sv.csv` inputs locally.

## Core Pipeline Scripts

- `echogram_processing.py`: single-dataset pipeline entrypoint (detection, review, metrics, exports).
- `dsl_tracking.py`: OpenCV contour detection, plotting helpers, CSV exports.
- `layer_review.py`: interactive contour review/edit UI.
- `batch_echogram_processing.py`: batch launcher for many `.sv.csv` files.
- `batch_master_config.py`: batch defaults and per-dataset params resolution.
- `run_layer_velocity_rollout_batch.py`: headless rerun pipeline using reviewed contours.
- `layer_velocity_rollout_config.py`: rollout dataset manifest and rollout path settings.
- `cross_dataset_dvm_summary.py`: aggregate rollout outputs into cross-dataset CSVs/figures.

## Python Requirements

Core dependencies:

- `numpy`
- `pandas`
- `matplotlib`
- `opencv-python`

Optional dependency:

- `scipy` (used for enhanced resampling helpers; pipeline still runs without it)
- `pywavelets` (used for optional wavelet denoising before contour detection)

Install from file:

```bash
python -m pip install -r requirements.txt
```

Or install directly:

```bash
python -m pip install numpy pandas matplotlib opencv-python scipy pywavelets
```

## Workflow 1: Run One Dataset

Use a params file:

```bash
ECHOGRAM_PARAMS_FILE="path/to/params_dataset.py" python echogram_processing.py
```

Optional module-based load:

```bash
ECHOGRAM_PARAMS_MODULE="params_B082D_CTD255" python echogram_processing.py
```

Common runtime overrides:

- `ECHOGRAM_PARAMS_FILE`
- `ECHOGRAM_PARAMS_MODULE`
- `ECHOGRAM_INPUT_FILE`
- `ECHOGRAM_DATASET_NAME`
- `ECHOGRAM_CRUISE_NAME`
- `ECHOGRAM_FIGURES_DIR`
- `ECHOGRAM_TIME_AXIS_MODE`

Review/runtime toggles:

- `ECHOGRAM_ENABLE_LAYER_REVIEW`
- `ECHOGRAM_SAVE_REVIEWED_CONTOURS`
- `ECHOGRAM_LOAD_REVIEWED_CONTOURS`
- `ECHOGRAM_SKIP_LAYER_REVIEW_IF_LOADED`
- `ECHOGRAM_REVIEWED_CONTOURS_FILENAME`
- `ECHOGRAM_REVIEWED_CONTOURS_SUBDIR`
- `ECHOGRAM_REVIEWED_CONTOURS_DIR`

Wavelet preprocessing toggles:

- `ECHOGRAM_WAVELET_ENABLE`
- `ECHOGRAM_WAVELET_NAME` (e.g., `db4`, `bior4.4`)
- `ECHOGRAM_WAVELET_LEVELS`
- `ECHOGRAM_WAVELET_THRESHOLD_MODE` (`soft`/`hard`)
- `ECHOGRAM_WAVELET_THRESHOLD_SCALE`
- `ECHOGRAM_WAVELET_CLIP_MIN`
- `ECHOGRAM_WAVELET_CLIP_MAX`
- `ECHOGRAM_WAVELET_APPLY_SECOND_PASS`

### Layer Review Features

Interactive review behavior is controlled by params:

- `ENABLE_LAYER_REVIEW`
- `LAYER_REVIEW_SCOPE` (`"main"`, `"all"`, `"choose_each_run"`)
- `LAYER_REVIEW_OUTPUT_SUBDIR`

Keyboard controls:

- `click`: select/unselect contour
- `m`: merge selected contours
- `d`: delete selected contours
- `r`: reset to original detected contours
- `u`: undo last contour-edit action
- `x`: start split mode (one selected contour)
- split mode: `click` points, `Enter` apply, `Backspace` remove last point, `c` cancel
- `f`: start bridge/fill mode (between two clicked points)
- bridge mode: `click` two points, `Enter` apply, `Backspace` remove last point, `c` cancel
- `s`: save reviewed contours and continue pipeline
- `q`: quit review without applying edits
- `Esc`: cancel split/bridge mode, otherwise quit without applying edits

### Reviewed Contour Persistence and Reuse

Params controlling artifact save/load:

- `SAVE_REVIEWED_CONTOURS`
- `LOAD_REVIEWED_CONTOURS`
- `SKIP_LAYER_REVIEW_IF_LOADED`
- `REVIEWED_CONTOURS_FILENAME` (default: `reviewed_contours.npz`)
- `REVIEWED_CONTOURS_SUBDIR` (default: `None`, resolves to reviewed folder)

Default artifact location:

- `Figures/<Cruise>/<Dataset>/<Dataset>_reviewed/reviewed_contours.npz`

Behavior:

1. First reviewed run (artifact missing):
   - review UI opens (if enabled)
   - press `s` to accept edits
   - artifact is written when save is enabled
2. Re-run with valid artifact:
   - contours load from artifact
   - review UI can be skipped
   - plots/metrics are recomputed from loaded contours

If artifact metadata does not match run context (dataset, ping/depth ranges, image shapes), the load is rejected and the pipeline falls back safely.

## Workflow 2: Batch Processing

Run:

```bash
python batch_echogram_processing.py
```

Batch behavior is controlled by `batch_master_config.py`.

Batch env overrides:

- `ECHOGRAM_BATCH_INPUT_CSV_DIR` (default: `EV DVM Echograms`)
- `ECHOGRAM_BATCH_OUTPUT_ROOT_DIR` (default: `Batch Results`)
- `ECHOGRAM_BATCH_EXPERIMENT_LABEL` (default: `ev_dvm_echograms_v1`)
- `ECHOGRAM_BATCH_PARAMS_ROOT` (default: `EV DVM Params`)

Default batch output structure:

- `Batch Results/<EXPERIMENT_LABEL>/<CRUISE>/<DATASET>/...`

## Workflow 3: Layer-Velocity Rollout + Cross-Dataset Summary

1) Headless rollout over configured datasets:

```bash
python run_layer_velocity_rollout_batch.py
```

2) Cross-dataset aggregation/plots:

```bash
python cross_dataset_dvm_summary.py
```

Rollout behavior is controlled by `layer_velocity_rollout_config.py`.

Rollout env overrides:

- `ECHOGRAM_ROLLOUT_OUTPUT_ROOT_DIR` (default: `Batch Results`)
- `ECHOGRAM_ROLLOUT_EXPERIMENT_LABEL` (default: `layer_velocity_rollout_v1`)
- `ECHOGRAM_ROLLOUT_EV_PARAMS_DIR` (default: `EV DVM Params`)
- `ECHOGRAM_ROLLOUT_DP09_PARAMS_DIR` (default: `DP09_CTD_params`)
- `ECHOGRAM_ROLLOUT_REVIEWED_ROOT` (default: `Figures`)

Notes:

- Rollout runs `echogram_processing.py` in headless review mode (review UI disabled).
- Rollout expects reviewed contour artifacts when `LOAD_REVIEWED_CONTOURS` is enabled for reproducible reruns.

## Output Reference

### Per-Dataset Outputs (`echogram_processing.py`)

Common outputs include:

- `echogram.png`
- `echogram_with_main_dsl.png`
- `echogram_with_diffuse_dsl.png` (when second pass is enabled)
- `echogram_with_all_dsl.png`
- `echogram_and_layer_speed.png`
- `echogram_and_main_layer_speed.png`
- `echogram_and_diffuse_layer_speed.png` (when diffuse layers are present)
- `dsl_layer_speed_vertical_summary.csv`
- `dsl_layer_speed_method_comparison.csv`

Review-related outputs:

- `layer_review_manifest.json`
- `reviewed_contours.npz` (when save is enabled)

Optional boolean CSV exports (when enabled):

- `<DATASET>_main_dsl_layers_boolean_csv/`
- `<DATASET>_diffuse_dsl_layers_boolean_csv/`

### Cross-Dataset Summary Outputs (`cross_dataset_dvm_summary.py`)

Generated under:

- `Batch Results/<ROLLOUT_EXPERIMENT_LABEL>/cross_dataset_summary/`

Typical outputs:

- `dataset_input_status.csv`
- `combined_main_layer_records.csv`
- `all_metric_aggregates.csv`
- `speed_method_long_records.csv`
- `speed_method_summary.csv`
- `speed_method_comparison_by_direction.png`
- `<metric>_by_direction_layer.csv`
- `<metric>_by_direction_layer.png`

