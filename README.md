# OpenCV Layer Tracking

Python pipeline for detecting DSL/DVM layers in echogram CSV exports, reviewing contours interactively, and producing plots/metrics/CSV outputs.

## Run A Dataset

Use a params file:

```bash
ECHOGRAM_PARAMS_FILE="EV DVM Params/DP03/params_DP03_B252D_38kHz.py" python echogram_processing.py
```

Optional module-based load:

```bash
ECHOGRAM_PARAMS_MODULE="params_B082D_CTD255" python echogram_processing.py
```

## Layer Review Features

Interactive review is controlled by params:

- `ENABLE_LAYER_REVIEW`
- `LAYER_REVIEW_SCOPE` (`"main"`, `"all"`, `"choose_each_run"`)
- `LAYER_REVIEW_OUTPUT_SUBDIR`

### Keyboard Controls

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

## Reviewed Contour Persistence And Reuse

These params control reviewed contour artifact save/load:

- `SAVE_REVIEWED_CONTOURS`
- `LOAD_REVIEWED_CONTOURS`
- `SKIP_LAYER_REVIEW_IF_LOADED`
- `REVIEWED_CONTOURS_FILENAME` (default: `"reviewed_contours.npz"`)
- `REVIEWED_CONTOURS_SUBDIR` (default: `None`, resolves to reviewed folder)

### Artifact Location

By default, reviewed geometry is saved in the reviewed output folder:

- `Figures/<Cruise>/<Dataset>/<Dataset>_reviewed/reviewed_contours.npz`

### First-Run vs Re-Run Behavior

1. First reviewed run (artifact does not exist):
   - interactive review opens (if enabled)
   - press `s` to accept edits
   - artifact is created if `SAVE_REVIEWED_CONTOURS=True`
2. Later reruns (artifact exists and metadata matches):
   - contours are loaded if `LOAD_REVIEWED_CONTOURS=True`
   - review UI is skipped when `SKIP_LAYER_REVIEW_IF_LOADED=True`
   - pipeline re-computes plots/metrics from loaded reviewed contours

If artifact metadata does not match current run context (dataset, ping/depth ranges, image shapes), load is safely rejected and the pipeline falls back to normal detection/review flow.

## Review Outputs

Typical reviewed output folder contents:

- `layer_review_manifest.json`
- `reviewed_contours.npz` (if save is enabled)
- `echogram_with_main_dsl.png`
- `echogram_with_all_dsl.png` (when applicable)
- `dsl_layer_speed_vertical_summary.csv`
- `echogram_and_layer_speed.png`
- `echogram_and_main_layer_speed.png`

## Environment Overrides (Common)

- `ECHOGRAM_PARAMS_FILE`
- `ECHOGRAM_PARAMS_MODULE`
- `ECHOGRAM_INPUT_FILE`
- `ECHOGRAM_DATASET_NAME`
- `ECHOGRAM_CRUISE_NAME`
- `ECHOGRAM_FIGURES_DIR`

Reviewed contour overrides:

- `ECHOGRAM_SAVE_REVIEWED_CONTOURS`
- `ECHOGRAM_LOAD_REVIEWED_CONTOURS`
- `ECHOGRAM_SKIP_LAYER_REVIEW_IF_LOADED`
- `ECHOGRAM_REVIEWED_CONTOURS_FILENAME`
- `ECHOGRAM_REVIEWED_CONTOURS_SUBDIR`
- `ECHOGRAM_REVIEWED_CONTOURS_DIR`

