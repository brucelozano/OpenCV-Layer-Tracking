import colorsys

import cv2
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np


def _get_distinct_colors(n):
    if n <= 0:
        return []

    colors = []
    for i in range(n):
        hue = i / max(n, 1)
        saturation = 0.9
        value = 0.9
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append((rgb[0], rgb[1], rgb[2]))
    return colors


def _contour_points_to_plot_space(
    points,
    img_width_pixels,
    img_height_pixels,
    min_ping,
    max_ping,
    depth_start,
    depth_stop,
    has_custom_x_axis,
    x_axis_values,
    x_pixel_positions,
    width_denominator,
    x_axis_mode,
    x_extent_start,
    x_extent_end,
):
    display_ping_range = max_ping - min_ping
    scaled_x = np.zeros(len(points), dtype=np.float64)
    scaled_y = np.zeros(len(points), dtype=np.float64)

    for i, (px, py) in enumerate(points):
        clamped_px = np.clip(px, 0, img_width_pixels - 1)
        if has_custom_x_axis:
            if x_axis_mode == "true_time":
                scaled_x[i] = np.interp(clamped_px, x_pixel_positions, x_axis_values)
            else:
                scaled_x[i] = x_extent_start + (
                    (clamped_px / width_denominator) * (x_extent_end - x_extent_start)
                )
        else:
            scaled_x[i] = min_ping + (clamped_px / width_denominator) * display_ping_range

        depth_fraction = py / max(img_height_pixels, 1)
        scaled_y[i] = depth_start + depth_fraction * (depth_stop - depth_start)

    return scaled_x, scaled_y


def _plot_x_to_pixel_x(
    x_value,
    min_ping,
    max_ping,
    has_custom_x_axis,
    x_axis_values,
    x_pixel_positions,
    width_denominator,
    x_axis_mode,
    x_extent_start,
    x_extent_end,
):
    if has_custom_x_axis:
        if x_axis_mode == "true_time":
            return float(np.interp(x_value, x_axis_values, x_pixel_positions))
        if x_extent_end == x_extent_start:
            return 0.0
        return float(((x_value - x_extent_start) / (x_extent_end - x_extent_start)) * width_denominator)

    if max_ping == min_ping:
        return 0.0
    return float(((x_value - min_ping) / (max_ping - min_ping)) * width_denominator)


def _plot_y_to_pixel_y(y_value, depth_start, depth_stop, img_height_pixels):
    if depth_stop == depth_start:
        return 0.0
    depth_fraction = (y_value - depth_start) / (depth_stop - depth_start)
    return float(depth_fraction * max(img_height_pixels - 1, 1))


def _compute_gradient_magnitude(image_data):
    if image_data is None:
        return None
    if np.asarray(image_data).ndim != 2:
        return None

    image_float = np.asarray(image_data, dtype=np.float32)
    if image_float.size == 0:
        return None

    grad_x = cv2.Sobel(image_float, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image_float, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(grad_x * grad_x + grad_y * grad_y)


def _densify_polyline(polyline_points, step_pixels=1.0):
    points = np.asarray(polyline_points, dtype=np.float64)
    if len(points) < 2:
        return points

    dense_points = [points[0]]
    for idx in range(1, len(points)):
        start = points[idx - 1]
        stop = points[idx]
        delta = stop - start
        seg_len = float(np.hypot(delta[0], delta[1]))
        if seg_len <= 1e-9:
            continue
        steps = max(1, int(np.ceil(seg_len / max(step_pixels, 1e-6))))
        for step_idx in range(1, steps + 1):
            t = step_idx / steps
            dense_points.append(start + (delta * t))
    return np.asarray(dense_points, dtype=np.float64)


def _build_rugged_path_mask(
    image_shape,
    polyline_points,
    base_thickness,
    amplitude,
    wavelength,
):
    mask = np.zeros(image_shape, dtype=np.uint8)
    dense_points = _densify_polyline(polyline_points, step_pixels=1.0)
    if len(dense_points) < 2:
        return mask

    deltas = np.diff(dense_points, axis=0)
    seg_lengths = np.hypot(deltas[:, 0], deltas[:, 1])
    cumulative = np.concatenate(([0.0], np.cumsum(seg_lengths)))

    centers = []
    for idx, point in enumerate(dense_points):
        prev_point = dense_points[max(0, idx - 1)]
        next_point = dense_points[min(len(dense_points) - 1, idx + 1)]
        tangent = next_point - prev_point
        tangent_norm = float(np.hypot(tangent[0], tangent[1]))
        if tangent_norm <= 1e-9:
            tangent = np.array([1.0, 0.0], dtype=np.float64)
            tangent_norm = 1.0
        tangent = tangent / tangent_norm
        normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)

        s = cumulative[idx]
        primary = np.sin((2.0 * np.pi * s) / max(wavelength, 1.0))
        secondary = np.sin((2.0 * np.pi * s) / max(wavelength * 0.57, 1.0) + 1.15)
        wobble = amplitude * (primary + 0.35 * secondary)

        center = point + (normal * wobble)
        center[0] = np.clip(center[0], 0, image_shape[1] - 1)
        center[1] = np.clip(center[1], 0, image_shape[0] - 1)
        centers.append(center)

        radius = max(1, int(round(base_thickness * 0.55)))
        center_int = tuple(np.round(center).astype(np.int32))
        cv2.circle(mask, center_int, radius, 255, thickness=cv2.FILLED)

    centers_int = np.round(np.asarray(centers)).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(
        mask,
        [centers_int],
        isClosed=False,
        color=255,
        thickness=max(1, int(base_thickness)),
        lineType=cv2.LINE_8,
    )
    return mask


def _build_merged_contour_from_selection(image_shape, selected_contours):
    """
    Merge selected contours with a middle-ground strategy:
    - retain original contour geometry where possible,
    - connect gaps using local convex-hull guidance,
    - avoid global over-rounding from a single hull over all points.
    """
    working_mask = np.zeros(image_shape, dtype=np.uint8)
    cv2.drawContours(working_mask, selected_contours, -1, 255, thickness=cv2.FILLED)

    max_iterations = 50
    for _ in range(max_iterations):
        component_contours, _ = cv2.findContours(
            working_mask.copy(),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )
        if len(component_contours) <= 1:
            break

        best_distance2 = None
        best_i = None
        best_j = None
        best_p1 = None
        best_p2 = None

        for i in range(len(component_contours)):
            contour_i_points = component_contours[i].reshape(-1, 2)
            sample_i_step = max(1, len(contour_i_points) // 400)
            sample_i = contour_i_points[::sample_i_step].astype(np.int32)
            if len(sample_i) == 0:
                continue

            for j in range(i + 1, len(component_contours)):
                contour_j_points = component_contours[j].reshape(-1, 2)
                sample_j_step = max(1, len(contour_j_points) // 400)
                sample_j = contour_j_points[::sample_j_step].astype(np.int32)
                if len(sample_j) == 0:
                    continue

                diff = (
                    sample_i[:, None, :].astype(np.int64)
                    - sample_j[None, :, :].astype(np.int64)
                )
                dist2 = np.sum(diff * diff, axis=2)
                min_flat_index = int(np.argmin(dist2))
                min_i, min_j = np.unravel_index(min_flat_index, dist2.shape)
                pair_distance2 = int(dist2[min_i, min_j])

                if best_distance2 is None or pair_distance2 < best_distance2:
                    best_distance2 = pair_distance2
                    best_i = i
                    best_j = j
                    best_p1 = tuple(int(v) for v in sample_i[min_i])  # (x, y)
                    best_p2 = tuple(int(v) for v in sample_j[min_j])  # (x, y)

        if best_p1 is None or best_p2 is None or best_i is None or best_j is None:
            return None

        # Use nearest component pair and local hull to define a guided connector.
        component_i = component_contours[best_i]
        component_j = component_contours[best_j]

        component_i_mask = np.zeros_like(working_mask)
        component_j_mask = np.zeros_like(working_mask)
        cv2.drawContours(component_i_mask, [component_i], -1, 255, thickness=cv2.FILLED)
        cv2.drawContours(component_j_mask, [component_j], -1, 255, thickness=cv2.FILLED)

        pair_points = np.vstack((component_i.reshape(-1, 2), component_j.reshape(-1, 2))).astype(np.int32)
        local_hull = cv2.convexHull(pair_points)
        local_hull_mask = np.zeros_like(working_mask)
        cv2.drawContours(local_hull_mask, [local_hull], -1, 255, thickness=cv2.FILLED)

        gap_distance = float(np.sqrt(max(best_distance2, 0)))
        dilation_radius = max(2, min(25, int(np.ceil(gap_distance / 2.0))))
        dilation_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * dilation_radius + 1, 2 * dilation_radius + 1),
        )
        dilated_i = cv2.dilate(component_i_mask, dilation_kernel, iterations=1)
        dilated_j = cv2.dilate(component_j_mask, dilation_kernel, iterations=1)
        bridge_zone = cv2.bitwise_and(dilated_i, dilated_j)

        connector_candidate = cv2.bitwise_and(local_hull_mask, bridge_zone)
        connector_candidate = cv2.bitwise_and(
            connector_candidate,
            cv2.bitwise_not(working_mask),
        )

        if cv2.countNonZero(connector_candidate) == 0:
            # Fallback keeps a guaranteed connection if constrained hull bridge is empty.
            line_thickness = max(2, min(15, int(np.ceil(gap_distance / 3.0))))
            cv2.line(working_mask, best_p1, best_p2, 255, thickness=line_thickness)
        else:
            working_mask = cv2.bitwise_or(working_mask, connector_candidate)

        # Light smoothing to avoid jagged bridge artifacts.
        working_mask = cv2.morphologyEx(
            working_mask,
            cv2.MORPH_CLOSE,
            np.ones((3, 3), dtype=np.uint8),
            iterations=1,
        )

    merged_contours, _ = cv2.findContours(
        working_mask.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not merged_contours:
        return None

    merged_contour = max(merged_contours, key=cv2.contourArea)
    if merged_contour is None or len(merged_contour) < 3:
        return None
    return merged_contour.astype(np.int32)


def _extend_polyline_endpoints(polyline_points, extension_length):
    points = np.asarray(polyline_points, dtype=np.float64)
    if len(points) < 2:
        return points

    extended_points = points.copy()

    start_vec = points[0] - points[1]
    start_norm = np.linalg.norm(start_vec)
    if start_norm > 0:
        extended_points[0] = points[0] + (start_vec / start_norm) * extension_length

    end_vec = points[-1] - points[-2]
    end_norm = np.linalg.norm(end_vec)
    if end_norm > 0:
        extended_points[-1] = points[-1] + (end_vec / end_norm) * extension_length

    return extended_points


def _split_contour_from_polyline(
    image_shape,
    contour,
    split_points_pixel,
    cut_thickness=3,
    image_data=None,
):
    """
    Split a contour using a user-drawn guide polyline.
    """
    if split_points_pixel is None or len(split_points_pixel) < 2:
        return None

    contour_mask = np.zeros(image_shape, dtype=np.uint8)
    cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)

    diag_len = float(np.hypot(image_shape[0], image_shape[1]))
    extended_points = _extend_polyline_endpoints(
        split_points_pixel,
        extension_length=diag_len,
    )

    perimeter = float(max(cv2.arcLength(contour, True), 1.0))
    area = float(max(cv2.contourArea(contour), 1.0))
    compactness = (perimeter * perimeter) / (4.0 * np.pi * area)
    rugged_amplitude = float(np.clip(1.0 + (compactness - 1.0) * 0.7, 1.0, 3.8))
    rugged_wavelength = float(np.clip(12.0 + 0.02 * perimeter, 10.0, 40.0))
    base_thickness = max(1, int(cut_thickness))

    cut_mask = _build_rugged_path_mask(
        image_shape=image_shape,
        polyline_points=extended_points,
        base_thickness=base_thickness,
        amplitude=rugged_amplitude,
        wavelength=rugged_wavelength,
    )

    gradient_magnitude = _compute_gradient_magnitude(image_data)
    if gradient_magnitude is not None:
        band_radius = max(3, min(30, int(base_thickness * 3)))
        band_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * band_radius + 1, 2 * band_radius + 1),
        )
        corridor_mask = cv2.dilate(cut_mask, band_kernel, iterations=1)
        corridor_mask = cv2.bitwise_and(corridor_mask, contour_mask)
        gradient_values = gradient_magnitude[corridor_mask > 0]
        if gradient_values.size > 0:
            gradient_threshold = float(np.percentile(gradient_values, 58))
            textured_cut = np.zeros_like(cut_mask)
            textured_cut[
                (gradient_magnitude >= gradient_threshold) & (corridor_mask > 0)
            ] = 255
            textured_cut = cv2.morphologyEx(
                textured_cut,
                cv2.MORPH_CLOSE,
                np.ones((3, 3), dtype=np.uint8),
                iterations=1,
            )
            textured_cut = cv2.bitwise_and(textured_cut, corridor_mask)
            cut_mask = cv2.bitwise_or(cut_mask, textured_cut)

    # Expand cut slightly so split succeeds with narrow paths.
    cut_mask = cv2.dilate(cut_mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
    split_candidate_mask = cv2.bitwise_and(contour_mask, cv2.bitwise_not(cut_mask))

    split_contours, _ = cv2.findContours(
        split_candidate_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    if len(split_contours) < 2:
        return None

    original_area = max(cv2.contourArea(contour), 1.0)
    min_piece_area = max(40.0, original_area * 0.003)
    valid_pieces = [
        piece.astype(np.int32)
        for piece in split_contours
        if cv2.contourArea(piece) >= min_piece_area
    ]
    if len(valid_pieces) < 2:
        return None

    valid_pieces.sort(key=cv2.contourArea, reverse=True)
    return valid_pieces


def _clip_point_to_image(point, image_shape):
    x = int(np.clip(round(float(point[0])), 0, image_shape[1] - 1))
    y = int(np.clip(round(float(point[1])), 0, image_shape[0] - 1))
    return x, y


def _find_contour_index_from_point(contours, point, outside_tolerance_px=8.0):
    if not contours:
        return None

    inside_indices = []
    best_signed_distance = -np.inf
    best_idx = None

    for idx, contour in enumerate(contours):
        signed_distance = float(
            cv2.pointPolygonTest(contour.astype(np.float32), point, True)
        )
        if signed_distance >= 0:
            inside_indices.append(idx)
        if signed_distance > best_signed_distance:
            best_signed_distance = signed_distance
            best_idx = idx

    if inside_indices:
        return min(inside_indices, key=lambda i: cv2.contourArea(contours[i]))

    if best_idx is not None and best_signed_distance >= -float(outside_tolerance_px):
        return int(best_idx)
    return None


def _points_share_component(binary_mask, point_a, point_b):
    if cv2.countNonZero(binary_mask) == 0:
        return False

    component_count, labels = cv2.connectedComponents(
        (binary_mask > 0).astype(np.uint8)
    )
    if component_count <= 1:
        return False

    ax, ay = _clip_point_to_image(point_a, binary_mask.shape)
    bx, by = _clip_point_to_image(point_b, binary_mask.shape)
    label_a = int(labels[ay, ax])
    label_b = int(labels[by, bx])
    return label_a > 0 and label_a == label_b


def _merge_or_fill_between_two_points(
    image_shape,
    contours,
    point_a_pixel,
    point_b_pixel,
):
    if len(contours) == 0:
        return None

    idx_a = _find_contour_index_from_point(contours, point_a_pixel)
    idx_b = _find_contour_index_from_point(contours, point_b_pixel)
    if idx_a is None or idx_b is None:
        return None

    source_indices = sorted(set([int(idx_a), int(idx_b)]))

    target_mask = np.zeros(image_shape, dtype=np.uint8)
    for idx in source_indices:
        cv2.drawContours(target_mask, [contours[idx]], -1, 255, thickness=cv2.FILLED)

    point_distance = float(
        np.hypot(
            point_a_pixel[0] - point_b_pixel[0],
            point_a_pixel[1] - point_b_pixel[1],
        )
    )
    base_thickness = max(2, min(16, int(np.ceil(point_distance / 8.0))))
    rugged_amplitude = float(np.clip(point_distance * 0.08, 0.8, 3.6))
    rugged_wavelength = float(np.clip(10.0 + point_distance * 0.9, 10.0, 38.0))
    path_mask = _build_rugged_path_mask(
        image_shape=image_shape,
        polyline_points=[point_a_pixel, point_b_pixel],
        base_thickness=base_thickness,
        amplitude=rugged_amplitude,
        wavelength=rugged_wavelength,
    )

    path_mask = cv2.dilate(path_mask, np.ones((3, 3), dtype=np.uint8), iterations=1)

    if len(source_indices) == 2:
        pair_points = np.vstack(
            (
                contours[source_indices[0]].reshape(-1, 2),
                contours[source_indices[1]].reshape(-1, 2),
            )
        ).astype(np.int32)
        pair_hull = cv2.convexHull(pair_points)
        hull_mask = np.zeros(image_shape, dtype=np.uint8)
        cv2.drawContours(hull_mask, [pair_hull], -1, 255, thickness=cv2.FILLED)
        bridge_mask = cv2.bitwise_and(path_mask, hull_mask)
        if cv2.countNonZero(bridge_mask) == 0:
            bridge_mask = path_mask
    else:
        support_radius = max(4, min(28, base_thickness * 3))
        support_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * support_radius + 1, 2 * support_radius + 1),
        )
        support_zone = cv2.dilate(target_mask, support_kernel, iterations=1)
        bridge_mask = cv2.bitwise_and(path_mask, support_zone)
        if cv2.countNonZero(bridge_mask) == 0:
            bridge_mask = path_mask

    combined_mask = target_mask.copy()
    for extra_dilate in range(0, 5):
        candidate = bridge_mask
        if extra_dilate > 0:
            candidate = cv2.dilate(
                candidate,
                np.ones((3, 3), dtype=np.uint8),
                iterations=extra_dilate,
            )
        trial_mask = cv2.bitwise_or(target_mask, candidate)
        trial_mask = cv2.morphologyEx(
            trial_mask,
            cv2.MORPH_CLOSE,
            np.ones((3, 3), dtype=np.uint8),
            iterations=1,
        )
        combined_mask = trial_mask
        if len(source_indices) < 2:
            break
        if _points_share_component(combined_mask, point_a_pixel, point_b_pixel):
            break

    merged_candidates, _ = cv2.findContours(
        combined_mask.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    if not merged_candidates:
        return None

    def _candidate_score(contour):
        contains_a = cv2.pointPolygonTest(contour.astype(np.float32), point_a_pixel, False) >= 0
        contains_b = cv2.pointPolygonTest(contour.astype(np.float32), point_b_pixel, False) >= 0
        score = int(contains_a) + int(contains_b)
        return score, float(cv2.contourArea(contour))

    chosen_contour = max(merged_candidates, key=_candidate_score)
    chosen_score, _ = _candidate_score(chosen_contour)
    if len(source_indices) == 2 and chosen_score < 2:
        return None

    updated_contours = [
        contour
        for idx, contour in enumerate(contours)
        if idx not in source_indices
    ]
    updated_contours.append(chosen_contour.astype(np.int32))

    new_index = len(updated_contours) - 1
    return {
        "contours": updated_contours,
        "source_indices": source_indices,
        "new_index": int(new_index),
    }


def review_contours_interactively(
    image_data,
    contours,
    depth_start,
    depth_stop,
    min_ping,
    max_ping,
    title="Interactive Layer Review",
    x_axis_values=None,
    x_axis_is_time=False,
    x_axis_label=None,
    x_axis_mode="true_time",
    layer_prefix="Layer",
    background_vmin=None,
    background_vmax=None,
    figure_size_inches=(9.5, 5.0),
    figure_dpi=110,
):
    """
    Interactive contour review session.

    Controls:
    - Click inside a contour: select/unselect.
    - m: merge selected contours into one.
    - f: bridge/fill between two clicked points.
    - bridge mode: click 2 points, Enter applies, Backspace removes last point, c cancels.
    - x: start split mode for one selected contour.
    - split mode: click to add cut points, Backspace removes last point.
    - split mode: Enter applies split, c cancels split mode.
    - d: delete selected contours.
    - r: reset to original contours.
    - u: undo the last contour-editing action.
    - s: save reviewed contours and continue.
    - Esc: cancel split/bridge mode (if active), otherwise quit review without applying changes.
    - q: quit review without applying changes.
    """
    original_contours = [contour.copy() for contour in contours]
    state = {
        "contours": [contour.copy() for contour in contours],
        "selected_indices": set(),
        "actions": [],
        "undo_stack": [],
        "accepted": False,
        "done": False,
        "split_mode": False,
        "split_target_index": None,
        "split_points_plot": [],
        "bridge_mode": False,
        "bridge_points_plot": [],
    }

    if len(state["contours"]) == 0:
        print("Layer review: no contours available to review.")
        return {
            "accepted": False,
            "contours": original_contours,
            "actions": [],
            "initial_count": 0,
            "final_count": 0,
        }

    original_keymap_save = list(plt.rcParams.get("keymap.save", []))
    original_keymap_fullscreen = list(plt.rcParams.get("keymap.fullscreen", []))
    plt.rcParams["keymap.save"] = []
    plt.rcParams["keymap.fullscreen"] = []
    try:
        image_shape = image_data.shape
        img_height_pixels = image_shape[0]
        img_width_pixels = image_shape[1]
        x_pixel_positions = np.arange(img_width_pixels, dtype=np.float64)
        width_denominator = max(img_width_pixels - 1, 1)
        axis_mode = str(x_axis_mode).strip().lower() if x_axis_mode is not None else "true_time"
        if axis_mode not in {"true_time", "linear"}:
            print(
                f"Warning: x_axis_mode={x_axis_mode!r} is invalid; "
                "using 'true_time' (supported: 'true_time', 'linear')."
            )
            axis_mode = "true_time"
        has_custom_x_axis = x_axis_values is not None and len(x_axis_values) == img_width_pixels
        if has_custom_x_axis:
            x_axis_values = np.asarray(x_axis_values, dtype=np.float64)
            if np.any(~np.isfinite(x_axis_values)) or np.any(np.diff(x_axis_values) <= 0):
                print(
                    "Warning: x_axis_values must be finite and strictly increasing; "
                    "falling back to ping-index x-axis for layer review."
                )
                has_custom_x_axis = False

        if has_custom_x_axis:
            x_extent_start = float(x_axis_values[0])
            x_extent_end = float(x_axis_values[-1])
        else:
            x_extent_start = min_ping
            x_extent_end = max_ping

        use_true_time_mesh = has_custom_x_axis and axis_mode == "true_time"
        if use_true_time_mesh:
            if img_width_pixels == 1:
                half_step = (1.0 / 86400.0) if x_axis_is_time else 0.5
                x_edges = np.array(
                    [x_axis_values[0] - half_step, x_axis_values[0] + half_step],
                    dtype=np.float64,
                )
            else:
                x_edges = np.empty(img_width_pixels + 1, dtype=np.float64)
                x_edges[1:-1] = 0.5 * (x_axis_values[:-1] + x_axis_values[1:])
                x_edges[0] = x_axis_values[0] - 0.5 * (x_axis_values[1] - x_axis_values[0])
                x_edges[-1] = x_axis_values[-1] + 0.5 * (x_axis_values[-1] - x_axis_values[-2])
        else:
            x_edges = None
        y_edges = np.linspace(depth_start, depth_stop, img_height_pixels + 1, dtype=np.float64)

        if x_axis_label is None:
            x_axis_label = "Time (UTC)" if x_axis_is_time else "Ping Number"

        figure, ax = plt.subplots(figsize=figure_size_inches, dpi=figure_dpi, constrained_layout=True)

        def redraw():
            ax.clear()
            if use_true_time_mesh and x_edges is not None:
                ax.pcolormesh(
                    x_edges,
                    y_edges,
                    image_data,
                    shading="auto",
                    cmap="viridis",
                    vmin=background_vmin,
                    vmax=background_vmax,
                )
                ax.set_ylim(depth_stop, depth_start)
                ax.set_xlim(float(x_edges[0]), float(x_edges[-1]))
            else:
                ax.imshow(
                    image_data,
                    aspect="auto",
                    cmap="viridis",
                    extent=[x_extent_start, x_extent_end, depth_stop, depth_start],
                    vmin=background_vmin,
                    vmax=background_vmax,
                )

            colors = _get_distinct_colors(len(state["contours"]))
            for idx, contour in enumerate(state["contours"]):
                points = contour.reshape(-1, 2)
                scaled_x, scaled_y = _contour_points_to_plot_space(
                    points=points,
                    img_width_pixels=img_width_pixels,
                    img_height_pixels=img_height_pixels,
                    min_ping=min_ping,
                    max_ping=max_ping,
                    depth_start=depth_start,
                    depth_stop=depth_stop,
                    has_custom_x_axis=has_custom_x_axis,
                    x_axis_values=x_axis_values,
                    x_pixel_positions=x_pixel_positions,
                    width_denominator=width_denominator,
                    x_axis_mode=axis_mode,
                    x_extent_start=x_extent_start,
                    x_extent_end=x_extent_end,
                )

                color = colors[idx]
                is_selected = idx in state["selected_indices"]
                face_alpha = 0.35 if is_selected else 0.22
                edge_color = "red" if is_selected else color
                line_width = 2.0 if is_selected else 1.0

                ax.fill(scaled_x, scaled_y, color=color, alpha=face_alpha)
                ax.plot(scaled_x, scaled_y, color=edge_color, linewidth=line_width)

                center_x = float(np.mean(scaled_x))
                center_y = float(np.mean(scaled_y))
                ax.text(
                    center_x,
                    center_y,
                    f"{idx + 1}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white",
                    bbox=dict(boxstyle="round", facecolor="black", alpha=0.65, pad=0.25),
                )

            ax.set_xlabel(x_axis_label)
            ax.set_ylabel("Depth (m)")
            ax.set_title(
                f"{title}\n"
                f"{layer_prefix} contours: {len(state['contours'])} | "
                f"Selected: {len(state['selected_indices'])}"
            )
            ax.grid(True, alpha=0.25)

            if state["split_mode"]:
                if state["split_points_plot"]:
                    split_x = [point[0] for point in state["split_points_plot"]]
                    split_y = [point[1] for point in state["split_points_plot"]]
                    ax.plot(
                        split_x,
                        split_y,
                        color="white",
                        linewidth=2.0,
                        linestyle="--",
                        marker="o",
                        markersize=4,
                    )
                ax.text(
                    0.01,
                    0.94,
                    "Split mode: click cut path points | Enter apply | Backspace undo | c cancel",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=9,
                    color="white",
                    bbox=dict(boxstyle="round", facecolor="black", alpha=0.6),
                )

            if state["bridge_mode"]:
                if state["bridge_points_plot"]:
                    bridge_x = [point[0] for point in state["bridge_points_plot"]]
                    bridge_y = [point[1] for point in state["bridge_points_plot"]]
                    ax.plot(
                        bridge_x,
                        bridge_y,
                        color="yellow",
                        linewidth=2.0,
                        linestyle="-.",
                        marker="o",
                        markersize=4,
                    )
                ax.text(
                    0.01,
                    0.88,
                    "Bridge mode: click 2 points | Enter apply | Backspace undo | c cancel",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=9,
                    color="black",
                    bbox=dict(boxstyle="round", facecolor="khaki", alpha=0.85),
                )

            if x_axis_is_time:
                locator = mdates.AutoDateLocator(minticks=4, maxticks=10)
                formatter = mdates.ConciseDateFormatter(locator)
                ax.xaxis.set_major_locator(locator)
                ax.xaxis.set_major_formatter(formatter)
                plt.setp(ax.get_xticklabels(), rotation=20, ha="right")

            help_text = (
                "Controls: click select | m merge | f bridge-fill | x split | d delete | r reset | u undo | s save | q quit"
            )
            ax.text(
                0.01,
                0.01,
                help_text,
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
            )

            figure.canvas.draw_idle()

        def toggle_selection_from_click(x_data, y_data):
            pixel_x = _plot_x_to_pixel_x(
                x_value=x_data,
                min_ping=min_ping,
                max_ping=max_ping,
                has_custom_x_axis=has_custom_x_axis,
                x_axis_values=x_axis_values,
                x_pixel_positions=x_pixel_positions,
                width_denominator=width_denominator,
                x_axis_mode=axis_mode,
                x_extent_start=x_extent_start,
                x_extent_end=x_extent_end,
            )
            pixel_y = _plot_y_to_pixel_y(
                y_value=y_data,
                depth_start=depth_start,
                depth_stop=depth_stop,
                img_height_pixels=img_height_pixels,
            )
            point = (float(pixel_x), float(pixel_y))

            containing_indices = []
            for idx, contour in enumerate(state["contours"]):
                inside = cv2.pointPolygonTest(contour.astype(np.float32), point, False)
                if inside >= 0:
                    containing_indices.append(idx)

            if not containing_indices:
                return

            chosen_idx = min(
                containing_indices,
                key=lambda i: cv2.contourArea(state["contours"][i]),
            )
            if chosen_idx in state["selected_indices"]:
                state["selected_indices"].remove(chosen_idx)
            else:
                state["selected_indices"].add(chosen_idx)

        def on_click(event):
            if event.inaxes != ax or event.xdata is None or event.ydata is None:
                return
            if state["split_mode"]:
                state["split_points_plot"].append((float(event.xdata), float(event.ydata)))
            elif state["bridge_mode"]:
                state["bridge_points_plot"].append((float(event.xdata), float(event.ydata)))
                if len(state["bridge_points_plot"]) > 2:
                    state["bridge_points_plot"] = state["bridge_points_plot"][-2:]
            else:
                toggle_selection_from_click(event.xdata, event.ydata)
            redraw()

        def cancel_bridge_mode():
            state["bridge_mode"] = False
            state["bridge_points_plot"] = []

        def cancel_split_mode():
            state["split_mode"] = False
            state["split_target_index"] = None
            state["split_points_plot"] = []

        def cancel_drawing_modes():
            cancel_split_mode()
            cancel_bridge_mode()

        def snapshot_before_edit(action_name, clear_drawing_modes=False):
            split_mode = False if clear_drawing_modes else bool(state["split_mode"])
            split_target_index = (
                None
                if clear_drawing_modes
                else (
                    int(state["split_target_index"])
                    if state["split_target_index"] is not None
                    else None
                )
            )
            split_points_plot = (
                []
                if clear_drawing_modes
                else [
                    (float(point[0]), float(point[1]))
                    for point in state["split_points_plot"]
                ]
            )
            bridge_mode = False if clear_drawing_modes else bool(state["bridge_mode"])
            bridge_points_plot = (
                []
                if clear_drawing_modes
                else [
                    (float(point[0]), float(point[1]))
                    for point in state["bridge_points_plot"]
                ]
            )

            state["undo_stack"].append(
                {
                    "action_name": str(action_name),
                    "contours": [contour.copy() for contour in state["contours"]],
                    "selected_indices": set(state["selected_indices"]),
                    "split_mode": split_mode,
                    "split_target_index": split_target_index,
                    "split_points_plot": split_points_plot,
                    "bridge_mode": bridge_mode,
                    "bridge_points_plot": bridge_points_plot,
                    "actions_len": len(state["actions"]),
                }
            )

        def restore_from_undo_snapshot(snapshot):
            state["contours"] = [contour.copy() for contour in snapshot["contours"]]
            state["selected_indices"] = set(snapshot["selected_indices"])
            state["split_mode"] = bool(snapshot["split_mode"])
            state["split_target_index"] = snapshot["split_target_index"]
            state["split_points_plot"] = [
                (float(point[0]), float(point[1]))
                for point in snapshot["split_points_plot"]
            ]
            state["bridge_mode"] = bool(snapshot["bridge_mode"])
            state["bridge_points_plot"] = [
                (float(point[0]), float(point[1]))
                for point in snapshot["bridge_points_plot"]
            ]

            actions_len = int(snapshot.get("actions_len", len(state["actions"])))
            state["actions"] = state["actions"][: max(actions_len, 0)]

        def undo_last_edit():
            if not state["undo_stack"]:
                print("Layer review: no previous edit to undo.")
                return

            snapshot = state["undo_stack"].pop()
            restore_from_undo_snapshot(snapshot)
            state["actions"].append(
                {"action": "undo", "undid": snapshot.get("action_name", "edit")}
            )
            print(f"Layer review: undid {snapshot.get('action_name', 'edit')}.")

        def start_bridge_mode():
            state["bridge_mode"] = True
            state["bridge_points_plot"] = []
            print("Layer review: bridge mode active. Click two points and press Enter to apply.")

        def apply_bridge_between_points():
            if not state["bridge_mode"]:
                return
            if len(state["bridge_points_plot"]) < 2:
                print("Layer review: add two bridge points before pressing Enter.")
                return

            point_a_plot, point_b_plot = state["bridge_points_plot"][:2]
            point_a_pixel = (
                _plot_x_to_pixel_x(
                    x_value=point_a_plot[0],
                    min_ping=min_ping,
                    max_ping=max_ping,
                    has_custom_x_axis=has_custom_x_axis,
                    x_axis_values=x_axis_values,
                    x_pixel_positions=x_pixel_positions,
                    width_denominator=width_denominator,
                    x_axis_mode=axis_mode,
                    x_extent_start=x_extent_start,
                    x_extent_end=x_extent_end,
                ),
                _plot_y_to_pixel_y(
                    y_value=point_a_plot[1],
                    depth_start=depth_start,
                    depth_stop=depth_stop,
                    img_height_pixels=img_height_pixels,
                ),
            )
            point_b_pixel = (
                _plot_x_to_pixel_x(
                    x_value=point_b_plot[0],
                    min_ping=min_ping,
                    max_ping=max_ping,
                    has_custom_x_axis=has_custom_x_axis,
                    x_axis_values=x_axis_values,
                    x_pixel_positions=x_pixel_positions,
                    width_denominator=width_denominator,
                    x_axis_mode=axis_mode,
                    x_extent_start=x_extent_start,
                    x_extent_end=x_extent_end,
                ),
                _plot_y_to_pixel_y(
                    y_value=point_b_plot[1],
                    depth_start=depth_start,
                    depth_stop=depth_stop,
                    img_height_pixels=img_height_pixels,
                ),
            )

            bridge_result = _merge_or_fill_between_two_points(
                image_shape=image_shape,
                contours=state["contours"],
                point_a_pixel=point_a_pixel,
                point_b_pixel=point_b_pixel,
            )
            if bridge_result is None:
                print(
                    "Layer review: bridge/fill failed. Click points on or near target contours."
                )
                return

            snapshot_before_edit("bridge_fill", clear_drawing_modes=True)
            state["contours"] = bridge_result["contours"]
            state["selected_indices"] = {bridge_result["new_index"]}
            state["actions"].append(
                {
                    "action": "bridge_fill",
                    "indices": bridge_result["source_indices"],
                    "new_index": bridge_result["new_index"],
                }
            )
            cancel_bridge_mode()

        def start_split_mode():
            if len(state["selected_indices"]) != 1:
                print("Layer review: select exactly one contour before splitting.")
                return
            target_idx = next(iter(state["selected_indices"]))
            if target_idx < 0 or target_idx >= len(state["contours"]):
                print("Layer review: selected contour index is invalid for split.")
                return
            state["split_mode"] = True
            state["split_target_index"] = int(target_idx)
            state["split_points_plot"] = []
            print("Layer review: split mode active. Click points and press Enter to apply.")

        def split_selected_contour():
            if not state["split_mode"]:
                return
            if len(state["split_points_plot"]) < 2:
                print("Layer review: add at least two split points before pressing Enter.")
                return

            target_idx = state["split_target_index"]
            if target_idx is None or target_idx < 0 or target_idx >= len(state["contours"]):
                print("Layer review: split target is no longer valid.")
                cancel_split_mode()
                return

            split_points_pixel = []
            for x_plot, y_plot in state["split_points_plot"]:
                pixel_x = _plot_x_to_pixel_x(
                    x_value=x_plot,
                    min_ping=min_ping,
                    max_ping=max_ping,
                    has_custom_x_axis=has_custom_x_axis,
                    x_axis_values=x_axis_values,
                    x_pixel_positions=x_pixel_positions,
                    width_denominator=width_denominator,
                    x_axis_mode=axis_mode,
                    x_extent_start=x_extent_start,
                    x_extent_end=x_extent_end,
                )
                pixel_y = _plot_y_to_pixel_y(
                    y_value=y_plot,
                    depth_start=depth_start,
                    depth_stop=depth_stop,
                    img_height_pixels=img_height_pixels,
                )
                split_points_pixel.append((pixel_x, pixel_y))

            target_contour = state["contours"][target_idx]
            split_pieces = _split_contour_from_polyline(
                image_shape=image_shape,
                contour=target_contour,
                split_points_pixel=split_points_pixel,
                cut_thickness=3,
                image_data=image_data,
            )
            if not split_pieces:
                print("Layer review: split failed. Draw a cut that crosses the contour.")
                return

            snapshot_before_edit("split", clear_drawing_modes=True)
            updated_contours = []
            for idx, contour in enumerate(state["contours"]):
                if idx == target_idx:
                    updated_contours.extend(split_pieces)
                else:
                    updated_contours.append(contour)
            state["contours"] = updated_contours
            state["actions"].append(
                {
                    "action": "split",
                    "index": int(target_idx),
                    "piece_count": int(len(split_pieces)),
                    "guide_points": int(len(state["split_points_plot"])),
                }
            )

            new_selection = set(
                range(
                    target_idx,
                    target_idx + len(split_pieces),
                )
            )
            state["selected_indices"] = new_selection
            cancel_split_mode()

        def remove_selected_contours():
            if not state["selected_indices"]:
                print("Layer review: no contours selected to remove.")
                return

            snapshot_before_edit("remove")
            cancel_drawing_modes()
            removed_indices = sorted(state["selected_indices"])
            state["contours"] = [
                contour
                for idx, contour in enumerate(state["contours"])
                if idx not in state["selected_indices"]
            ]
            state["actions"].append(
                {"action": "remove", "indices": removed_indices}
            )
            state["selected_indices"].clear()

        def merge_selected_contours():
            if len(state["selected_indices"]) < 2:
                print("Layer review: select at least two contours to merge.")
                return

            cancel_drawing_modes()
            merge_indices = sorted(state["selected_indices"])
            selected_contours = [state["contours"][idx] for idx in merge_indices]
            merged_contour = _build_merged_contour_from_selection(
                image_shape=image_shape,
                selected_contours=selected_contours,
            )
            if merged_contour is None:
                print("Layer review: merge failed (insufficient merged area).")
                return

            snapshot_before_edit("merge")
            remaining = [
                contour
                for idx, contour in enumerate(state["contours"])
                if idx not in state["selected_indices"]
            ]
            remaining.append(merged_contour)
            state["contours"] = remaining
            state["actions"].append(
                {"action": "merge", "indices": merge_indices, "new_index": len(remaining) - 1}
            )
            state["selected_indices"].clear()

        def reset_contours():
            snapshot_before_edit("reset")
            state["contours"] = [contour.copy() for contour in original_contours]
            state["selected_indices"].clear()
            cancel_drawing_modes()
            state["actions"].append({"action": "reset"})

        def finish(accepted):
            state["accepted"] = bool(accepted)
            state["done"] = True
            plt.close(figure)

        def on_key(event):
            key = (event.key or "").lower()
            in_split_mode = state["split_mode"]
            in_bridge_mode = state["bridge_mode"]
            in_drawing_mode = in_split_mode or in_bridge_mode

            if key == "q":
                finish(accepted=False)
            elif key == "escape":
                if in_drawing_mode:
                    cancel_drawing_modes()
                    redraw()
                else:
                    finish(accepted=False)
            elif key == "s":
                if in_split_mode:
                    print("Layer review: press Enter to apply split, or c/Esc to cancel split mode.")
                    return
                if in_bridge_mode:
                    print("Layer review: press Enter to apply bridge/fill, or c/Esc to cancel.")
                    return
                finish(accepted=True)
            elif key == "f":
                if in_split_mode:
                    print("Layer review: finish/cancel split mode before starting bridge mode.")
                    return
                if in_bridge_mode:
                    print("Layer review: bridge mode is already active.")
                else:
                    start_bridge_mode()
                redraw()
            elif key == "x":
                if in_bridge_mode:
                    print("Layer review: finish/cancel bridge mode before starting split mode.")
                    return
                if in_split_mode:
                    print("Layer review: split mode is already active.")
                else:
                    start_split_mode()
                redraw()
            elif key in ("enter", "return"):
                if in_split_mode:
                    split_selected_contour()
                    redraw()
                elif in_bridge_mode:
                    apply_bridge_between_points()
                    redraw()
            elif key == "backspace":
                if in_split_mode:
                    if state["split_points_plot"]:
                        state["split_points_plot"].pop()
                    redraw()
                elif in_bridge_mode:
                    if state["bridge_points_plot"]:
                        state["bridge_points_plot"].pop()
                    redraw()
            elif key == "c":
                if in_drawing_mode:
                    cancel_drawing_modes()
                    redraw()
            elif key == "u":
                undo_last_edit()
                redraw()
            elif key == "m":
                if in_drawing_mode:
                    print("Layer review: finish/cancel split or bridge mode before merging.")
                    return
                merge_selected_contours()
                redraw()
            elif key == "d":
                if in_drawing_mode:
                    print("Layer review: finish/cancel split or bridge mode before deleting.")
                    return
                remove_selected_contours()
                redraw()
            elif key == "r":
                if in_drawing_mode:
                    print("Layer review: finish/cancel split or bridge mode before resetting.")
                    return
                reset_contours()
                redraw()

        def on_close(_event):
            if not state["done"]:
                state["accepted"] = False
                state["done"] = True

        figure.canvas.mpl_connect("button_press_event", on_click)
        figure.canvas.mpl_connect("key_press_event", on_key)
        figure.canvas.mpl_connect("close_event", on_close)

        redraw()
        plt.show(block=True)
    finally:
        plt.rcParams["keymap.save"] = original_keymap_save
        plt.rcParams["keymap.fullscreen"] = original_keymap_fullscreen

    final_contours = state["contours"] if state["accepted"] else original_contours
    return {
        "accepted": state["accepted"],
        "contours": [contour.copy() for contour in final_contours],
        "actions": state["actions"],
        "initial_count": len(original_contours),
        "final_count": len(final_contours),
    }
