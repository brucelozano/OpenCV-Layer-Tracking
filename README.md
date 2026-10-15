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
- `sv_csv_loader.py`: shared `.sv.csv` loader with strict/ragged parsing modes.
- `run_layer_velocity_rollout_batch.py`: headless rerun pipeline using reviewed contours.
- `layer_velocity_rollout_config.py`: rollout dataset manifest and rollout path settings.
- `cross_dataset_dvm_summary.py`: aggregate rollout outputs into cross-dataset CSVs/figures.
- `layer_classification_workflow.py`: paired 18/38 kHz layer classification pipeline (delta Sv and composition).
- `layer_classification_config.py`: config for paired-frequency discovery, mask reuse/fallback, and class windows.
- `run_layer_classification_batch.py`: launcher for the layer classification workflow.

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

Windows shell equivalents for params-file runs:

PowerShell:

```powershell
$env:ECHOGRAM_PARAMS_FILE = "path/to/params_dataset.py"
python .\echogram_processing.py
```

Command Prompt (`cmd.exe`):

```cmd
set ECHOGRAM_PARAMS_FILE=path/to/params_dataset.py && python echogram_processing.py
```

### Running Test Fixtures (`test_params` + `test_csvs`)

If you keep lightweight test inputs in `test_csvs/` and matching params in
`test_params/`, run with a params file path:

```bash
ECHOGRAM_PARAMS_FILE="test_params/DP03/params_DP03_B082N_18kHz.py" ECHOGRAM_FIGURES_DIR="test_outputs" python echogram_processing.py
```

PowerShell equivalent:

```powershell
$env:ECHOGRAM_PARAMS_FILE = "test_params/DP03/params_DP03_B082N_18kHz.py"
$env:ECHOGRAM_FIGURES_DIR = "test_outputs"
python .\echogram_processing.py
```

Command Prompt (`cmd.exe`) equivalent:

```cmd
set ECHOGRAM_PARAMS_FILE=test_params/DP03/params_DP03_B082N_18kHz.py && set ECHOGRAM_FIGURES_DIR=test_outputs && python echogram_processing.py
```

Output path behavior:

- With cruise resolved: `test_outputs/<CRUISE>/<DATASET>/`
- Without cruise resolved: `test_outputs/<DATASET>/`

Notes:

- `ECHOGRAM_FIGURES_DIR` is treated as the output root.
- `test_outputs/` is intended for local generated artifacts and is git-ignored.

Common runtime overrides:

- `ECHOGRAM_PARAMS_FILE`
- `ECHOGRAM_PARAMS_MODULE`
- `ECHOGRAM_INPUT_FILE`
- `ECHOGRAM_DATASET_NAME`
- `ECHOGRAM_CRUISE_NAME`
- `ECHOGRAM_FIGURES_DIR`
- `ECHOGRAM_GROUP_OUTPUTS_BY_CRUISE`
- `ECHOGRAM_TIME_AXIS_MODE`
- `ECHOGRAM_SAMPLE_COLUMN_MODE` (`auto`, `strict`, `ragged`)

`ECHOGRAM_FIGURES_DIR` is treated as the output root (replacement for `./Figures`),
while cruise/dataset auto-folder logic still applies under that root.

`ECHOGRAM_SAMPLE_COLUMN_MODE` controls how `.sv.csv` sample columns are parsed:

- `auto` (default): tries strict fixed-width parsing first and falls back to ragged parsing
  if row widths vary.
- `strict`: preserves original parser behavior (fastest, requires stable row width).
- `ragged`: always uses row-wise parsing and validates each row against
  `fixed_columns + Sample_count`.

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

1. Headless rollout over configured datasets:

```bash
python run_layer_velocity_rollout_batch.py
```

1. Cross-dataset aggregation/plots:

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

## Workflow 4: 18/38 kHz Layer Classification

Run:

```bash
python run_layer_classification_batch.py
```

Or run directly:

```bash
python layer_classification_workflow.py
```

Classification behavior is controlled by `layer_classification_config.py`.

Expected input layout:

- Place paired-frequency `.sv.csv` files under `Layer Classification/` (or override via env var).
- Pairing is inferred from `18kHz` / `38kHz` tokens in filenames; matching base names form one pair.

Core processing:

- Align 18 and 38 kHz pings by nearest timestamp (default max delta: `2.5s`).
- Compute `delta_sv = sv_18khz - sv_38khz`.
- Reuse reviewed layer masks when available (optional tracking fallback can generate missing masks).
- Classify in-layer samples using configured windows:
  - `swim_bladdered_fish`: `[0, 12]`
  - `large_non_sb_fish`: `[-3, 0)`
  - `small_non_sb_fish_crustaceans`: `[-14, -3)`
  - `unclassified`: finite in-layer samples outside these ranges

Layer classification env overrides:

- `LAYER_CLASS_INPUT_ROOT_DIR` (default: `Layer Classification`)
- `LAYER_CLASS_OUTPUT_ROOT_DIR` (default: `Batch Results`)
- `LAYER_CLASS_EXPERIMENT_LABEL` (default: `layer_classification_v1`)
- `LAYER_CLASS_REVIEWED_ROOT` (default: `Figures`)
- `LAYER_CLASS_REUSE_REVIEWED_MASKS` (`1`/`0`)
- `LAYER_CLASS_RUN_TRACKING_IF_MASK_MISSING` (`1`/`0`)
- `LAYER_CLASS_MASK_SOURCE_KHZ` (default: `38`; supports `18`, `38`, `both`, `18,38`, `38,18`)
- `LAYER_CLASS_MAX_ALIGNMENT_DELTA_SECONDS` (default: `2.5`)
- `LAYER_CLASS_SAVE_DEBUG_PLOTS` (`1`/`0`, default: `0`)
- `LAYER_CLASS_DEBUG_PLOT_DPI` (default: `220`)
- `LAYER_CLASS_DEBUG_DELTA_VMIN` (default: `-15.0`)
- `LAYER_CLASS_DEBUG_DELTA_VMAX` (default: `12.0`)

Recommended run patterns:

Single-mask baseline (38 kHz mask source):

```bash
LAYER_CLASS_MASK_SOURCE_KHZ=38 LAYER_CLASS_SAVE_DEBUG_PLOTS=1 python run_layer_classification_batch.py
```

Dual-mask comparison (run both 38-mask and 18-mask scenarios in one pass):

```bash
LAYER_CLASS_MASK_SOURCE_KHZ=both LAYER_CLASS_SAVE_DEBUG_PLOTS=1 python run_layer_classification_batch.py
```

Interpretation notes:

- `delta_sv = Sv18 - Sv38` is computed the same way regardless of mask source.
- Final class percentages can still differ across runs because mask geometry changes which pixels are included.
- Use `layer_classification_alignment_diagnostics.csv` to confirm synchronization quality:
  - `matched`: whether a reference ping found a target ping within tolerance
  - `abs_time_delta_s`: nearest-neighbor time gap (0 means exact timestamp match)

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

### Layer Classification Outputs (`layer_classification_workflow.py`)

Generated under:

- `Batch Results/<LAYER_CLASS_EXPERIMENT_LABEL>/`

Top-level outputs:

- `layer_classification_layer_summary.csv`
- `layer_classification_dataset_summary.csv`
- `layer_classification_alignment_diagnostics.csv`
- `layer_classification_run_metadata.json`

Per-pair outputs:

- `Batch Results/<LAYER_CLASS_EXPERIMENT_LABEL>/<PAIR_KEY>/layer_classification_layer_summary.csv`
- `Batch Results/<LAYER_CLASS_EXPERIMENT_LABEL>/<PAIR_KEY>/layer_classification_dataset_summary.csv`
- `Batch Results/<LAYER_CLASS_EXPERIMENT_LABEL>/<PAIR_KEY>/layer_classification_alignment_diagnostics.csv`

When running with `LAYER_CLASS_MASK_SOURCE_KHZ=both` (or a two-value list), each pair has per-mask subfolders:

- `Batch Results/<LAYER_CLASS_EXPERIMENT_LABEL>/<PAIR_KEY>/mask_source_38khz/`
- `Batch Results/<LAYER_CLASS_EXPERIMENT_LABEL>/<PAIR_KEY>/mask_source_18khz/`
- `Batch Results/<LAYER_CLASS_EXPERIMENT_LABEL>/<PAIR_KEY>/debug_class_map_comparison_18_top_38_bottom.png` (when debug plots are enabled)

Optional per-pair debug outputs for single-mask runs (when `LAYER_CLASS_SAVE_DEBUG_PLOTS=1`):

- `Batch Results/<LAYER_CLASS_EXPERIMENT_LABEL>/<PAIR_KEY>/debug_plots/debug_aligned_18khz_sv.png`
- `Batch Results/<LAYER_CLASS_EXPERIMENT_LABEL>/<PAIR_KEY>/debug_plots/debug_aligned_38khz_sv.png`
- `Batch Results/<LAYER_CLASS_EXPERIMENT_LABEL>/<PAIR_KEY>/debug_plots/debug_mask_source_with_reviewed_contour.png`
- `Batch Results/<LAYER_CLASS_EXPERIMENT_LABEL>/<PAIR_KEY>/debug_plots/debug_delta_sv_full.png`
- `Batch Results/<LAYER_CLASS_EXPERIMENT_LABEL>/<PAIR_KEY>/debug_plots/debug_delta_sv_masked.png`
- `Batch Results/<LAYER_CLASS_EXPERIMENT_LABEL>/<PAIR_KEY>/debug_plots/debug_class_map.png`
- `Batch Results/<LAYER_CLASS_EXPERIMENT_LABEL>/<PAIR_KEY>/debug_plots/debug_layer_<N>_delta_histogram.png`

In dual-mask mode, the same debug filenames are written under each mask-specific folder:

- `Batch Results/<LAYER_CLASS_EXPERIMENT_LABEL>/<PAIR_KEY>/mask_source_38khz/debug_plots/`
- `Batch Results/<LAYER_CLASS_EXPERIMENT_LABEL>/<PAIR_KEY>/mask_source_18khz/debug_plots/`

### Layer Classification CSV Column Guide

`layer_classification_layer_summary.csv` (one row per detected layer):

- `pair_key`: paired dataset identifier (derived from filename base).
- `layer_index`: 1-based index of the layer contour within that pair/mask scenario.
- `masked_sample_count`: total pixels inside this layer mask.
- `finite_delta_count`: in-mask pixels where `delta_sv` is finite.
- `missing_delta_count`: in-mask pixels with missing/invalid `delta_sv`.
- `matched_reference_pings` / `unmatched_reference_pings`: ping-time alignment counts used for this row.
- `dataset_18`, `dataset_38`: dataset names used for subtraction.
- `mask_source_dataset`, `mask_source_frequency_khz`: which reviewed contour source defined the mask geometry.
- `reviewed_contours_artifact`: path to the `.npz` reviewed contour file used.

`layer_classification_dataset_summary.csv` (one row per pair+mask scenario):

- `layer_count`: number of layer rows aggregated into this summary row.
- `alignment_reference_ping_count`, `alignment_matched_ping_count`, `alignment_unmatched_ping_count`, `alignment_match_percent`: timestamp alignment quality rollup.
- `total_masked_sample_count`, `total_finite_delta_count`, `total_missing_delta_count`: aggregate in-mask sample totals across layers.
- `cruise_name`: inferred cruise folder/tag when available.
- All dataset/mask-source columns above are included here as context too.

Class metric columns (present in both layer and dataset summaries):

- `<class>_count`: sample count assigned to that class.
- `<class>_percent`: percent of finite in-mask samples (`finite_delta_count` for layer rows; `total_finite_delta_count` for dataset rows).
- `<class>_mean_delta_sv`, `<class>_median_delta_sv`: summary stats of `delta_sv` values in that class.
- Includes configured classes plus `unclassified`.

`layer_classification_alignment_diagnostics.csv` (one row per reference ping):

- `reference_dataset`, `target_dataset`: which dataset provided the reference grid and which was aligned to it.
- `mask_source_frequency_khz`: mask scenario for this row (`18` or `38` in dual-mask mode).
- `reference_ping_index`, `reference_time_utc`: reference ping identity/time.
- `matched`, `matched_ping_index`, `matched_time_utc`, `abs_time_delta_s`: nearest-neighbor alignment result per ping.
- `tolerance_s`: maximum allowed alignment delta used for matching.

Dual-mask note:

- With `LAYER_CLASS_MASK_SOURCE_KHZ=both`, you will see multiple rows with the same `pair_key`; use `mask_source_frequency_khz` to separate 18-mask vs 38-mask results.

