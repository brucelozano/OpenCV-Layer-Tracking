import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
from collections import defaultdict
import colorsys
import pandas as pd

try:
    from scipy import ndimage
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not available. Enhanced resampling features disabled.")

def save_debug_image(image_data, title, filename, figures_dir, cmap='gray', vmin=None, vmax=None):
    """Helper function to save debug images.
    
    Args:
        image_data: Image data to save
        title: Title for the plot
        filename: Filename to save as
        figures_dir: Directory to save in
        cmap: Colormap to use (default: gray)
    """
    plt.figure(figsize=(10, 7))
    plt.imshow(image_data, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.colorbar()
    save_path = os.path.join(figures_dir, filename)
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved debug image: {save_path}")

def preprocess_for_opencv(sv_data, dsl_sv_threshold_min, dsl_sv_threshold_max):
    """
    Prepares Sv data for OpenCV processing by normalizing to 0-255 range.
    
    Args:
        sv_data: Original Sv data array
        dsl_sv_threshold_min: Minimum Sv value for thresholding
        dsl_sv_threshold_max: Maximum Sv value for thresholding
        
    Returns:
        img_8bit: 8-bit image suitable for OpenCV processing
    """
    # Ensure sv_data is a NumPy array
    sv_data_np = np.array(sv_data, dtype=np.float32)
    
    # Handle NaN or infinite values
    sv_data_np = np.nan_to_num(sv_data_np, nan=dsl_sv_threshold_min - 10)
    
    # Normalize to 0-255 using the DSL threshold range

    # Provides some "breathing room" or visual separation 
    # If values slightly outside our exact DSL thresholds are 
    # still of interest for visualization or some types 
    # of processing, this margin ensures they aren't immediately clipped 
    # to pure black or pure white in the 8-bit image.

    min_val = dsl_sv_threshold_min - 5  # Add some margin
    max_val = dsl_sv_threshold_max + 5  # Add some margin
    
    if max_val == min_val:  # Avoid division by zero if data is flat
        normalized_data = np.zeros_like(sv_data_np, dtype=np.uint8)
    else:
        normalized_data = ((sv_data_np - min_val) / (max_val - min_val)) * 255.0
    
    # Clip values to be safe, and convert to uint8
    normalized_data = np.clip(normalized_data, 0, 255)
    img_8bit = normalized_data.astype(np.uint8)
    
    return img_8bit

def detect_dsl_contours(
    image_8bit_ignored, sv_data_original, 
    # Pass parameters
    sv_threshold_min,
    sv_threshold_max,
    min_contour_area,
    morph_kernel_size,
    morph_close_iterations,
    morph_open_iterations,
    contour_epsilon_factor,
    debug_dvm_vmin,
    debug_dvm_vmax,
    figures_dir_for_debug=None,
    stage_prefix=""  # To differentiate debug files
):
    """
    Detects DSL contours in the original Sv data using provided thresholds.
    Saves intermediate debug images.
    
    Args:
        image_8bit_ignored: Previously for an 8-bit image, now ignored as we operate on sv_data_original
        sv_data_original: Original Sv data array (already cropped to ROI) for thresholding
        figures_dir_for_debug: Directory to save debug images
        sv_threshold_min: Minimum Sv value for thresholding
        sv_threshold_max: Maximum Sv value for thresholding
        min_contour_area: Minimum pixel area for valid contours
        morph_kernel_size: Size of morphological operation kernel
        morph_close_iterations: Number of closing iterations
        morph_open_iterations: Number of opening iterations
        contour_epsilon_factor: Factor for contour approximation
        debug_dvm_vmin: Minimum value for debug image visualization
        debug_dvm_vmax: Maximum value for debug image visualization
        stage_prefix: Prefix for debug image filenames
        
    Returns:
        tuple: (filtered_contours, processed_mask)
    """
    print(f"\n--- Debugging detect_dsl_contours ({stage_prefix}) ---")
    print(f"Input sv_data_original shape: {sv_data_original.shape}")
    print(f"Using thresholds: Min Sv={sv_threshold_min}, Max Sv={sv_threshold_max}")
    
    if figures_dir_for_debug:
        debug_dir = os.path.join(figures_dir_for_debug, "dsl_debug_stages")
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir)
            
        # Stage 0: Visualize the input sv_data_original
        save_debug_image(sv_data_original,
                        f'Stage 0: Input Sv Data ({stage_prefix}Shape: {sv_data_original.shape})',
                        f'{stage_prefix}0_input_sv_data.png',
                        debug_dir,
                        cmap='viridis',
                        vmin=debug_dvm_vmin,
                        vmax=debug_dvm_vmax)
    
    # Stage 1: Thresholding based on original Sv values
    dsl_mask = ((sv_data_original >= sv_threshold_min) & 
                (sv_data_original <= sv_threshold_max)).astype(np.uint8) * 255
    
    if figures_dir_for_debug:
        save_debug_image(dsl_mask,
                        f'Stage 1: Initial Thresholded Mask ({stage_prefix}Sv: {sv_threshold_min} to {sv_threshold_max})',
                        f'{stage_prefix}1_thresholded_mask.png',
                        debug_dir)
    
    # Stage 2: Apply morphological operations to smooth the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, morph_kernel_size)
    
    # Close small gaps (dilate then erode)
    processed_mask = cv2.morphologyEx(dsl_mask, 
                                    cv2.MORPH_CLOSE, 
                                    kernel,
                                    iterations=morph_close_iterations)
    
    # Remove small noise (erode then dilate)
    processed_mask = cv2.morphologyEx(processed_mask, 
                                    cv2.MORPH_OPEN, 
                                    kernel,
                                    iterations=morph_open_iterations)
    
    if figures_dir_for_debug:
        save_debug_image(processed_mask,
                        f'Stage 2: {stage_prefix}Processed Mask After Morphological Operations\n' +
                        f'(Close iterations: {morph_close_iterations}, ' +
                        f'Open iterations: {morph_open_iterations})',
                        f'{stage_prefix}2_processed_mask.png',
                        debug_dir)
    
    # Stage 3: Initial Contour Detection
    contours, _ = cv2.findContours(processed_mask.copy(),
                                  cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)
    print(f"{stage_prefix}Found {len(contours)} raw contours.")
    
    if figures_dir_for_debug:
        # Visualize raw contours on mask
        contour_viz_raw = cv2.cvtColor(processed_mask, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(contour_viz_raw, contours, -1, (0, 0, 255), 1)
        save_debug_image(contour_viz_raw,
                        f'Stage 3: {stage_prefix}Raw Contours Before Filtering',
                        f'{stage_prefix}3_raw_contours_on_mask.png',
                        debug_dir)
    
    # Stage 4: Filter and Smooth Contours
    filtered_contours = []
    if contours:
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > min_contour_area:
                # Smooth the contour using approxPolyDP with parameter from params.py
                epsilon = contour_epsilon_factor * cv2.arcLength(contour, True)
                smoothed_contour = cv2.approxPolyDP(contour, epsilon, True)
                filtered_contours.append(smoothed_contour)
            else:
                #print(f"Contour {i} with area {area:.1f} rejected (min area: {min_contour_area}).")
                pass
    print(f"{stage_prefix}Kept {len(filtered_contours)} filtered contours (Min Area: {min_contour_area}).")
    
    if figures_dir_for_debug:
        # Stage 4: Visualize filtered contours on mask
        contour_viz_filtered = cv2.cvtColor(processed_mask, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(contour_viz_filtered, filtered_contours, -1, (0, 255, 0), 1)
        save_debug_image(contour_viz_filtered,
                        f'Stage 4: {stage_prefix}Filtered and Smoothed Contours',
                        f'{stage_prefix}4_filtered_contours_on_mask.png',
                        debug_dir)
        
        # Stage 5: Final visualization on original data
        sv_display = sv_data_original.copy()
        min_sv, max_sv = np.min(sv_display), np.max(sv_display)
        if max_sv > min_sv:
            sv_display = (sv_display - min_sv) / (max_sv - min_sv) * 255
        sv_display = sv_display.astype(np.uint8)
        sv_display_bgr = cv2.cvtColor(sv_display, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(sv_display_bgr, filtered_contours, -1, (0, 255, 0), 1)
        save_debug_image(sv_display_bgr,
                        f'Stage 5: {stage_prefix}Final Filtered Contours on Original Data',
                        f'{stage_prefix}5_filtered_contours_on_sv_data.png',
                        debug_dir)
    
    print(f"--- End Debugging detect_dsl_contours ({stage_prefix}) ---\n")
    return filtered_contours, processed_mask

def get_distinct_colors(n):
    """
    Generate n visually distinct colors using HSV color space.
    Returns a list of hex color codes.
    
    Args:
        n (int): Number of distinct colors to generate
        
    Returns:
        list: List of hex color codes
    """
    colors = []
    for i in range(n):
        # Distribute hues evenly around the color wheel
        hue = i / n
        # Use high saturation and value for vivid colors
        saturation = 0.9
        value = 0.9
        
        # Convert HSV to RGB (values between 0 and 1)
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        
        # Convert RGB values to hex color code
        hex_color = '#{:02x}{:02x}{:02x}'.format(
            int(rgb[0] * 255),
            int(rgb[1] * 255),
            int(rgb[2] * 255)
        )
        colors.append(hex_color)
    
    return colors

def plot_echogram_with_dsl(image_data, depth_start, depth_stop, min_ping, max_ping,
                          contours, vmin, vmax, title='Echogram with DVM Tracks', 
                          save_path=None, figures_dir='Figures', show_outline=True,
                          contour_info=None,
                          x_axis_values=None,
                          x_axis_is_time=False,
                          x_axis_label=None,
                          x_axis_mode='true_time'):  # [(contours, colors, label), ...]
    """
    Plots the echogram and overlays detected DVM contours.
    
    Args:
        image_data: Original Sv data array
        depth_start: Starting depth for display
        depth_stop: Stopping depth for display
        min_ping: Minimum ping number
        max_ping: Maximum ping number
        contours: List of detected contours
        vmin: Minimum value for color scaling
        vmax: Maximum value for color scaling
        title: Plot title
        save_path: Path to save the figure
        figures_dir: Directory to save figures
        show_outline: Whether to show the contour outline
        contour_info: List of tuples [(contours, colors, label), ...] for different layer types
        x_axis_values: Optional x-axis values per image column (len == image width)
        x_axis_is_time: Whether x-axis values represent Matplotlib datetime numbers
        x_axis_label: Custom x-axis label
        x_axis_mode: "true_time" (per-ping mapping) or "linear" (start/end interpolation)
    """
    # Create figure with extra space on right for colorbar and legend
    plt.figure(figsize=(16, 8))
    
    # Create main axis for echogram
    ax = plt.gca()
    
    # Get image dimensions for coordinate mapping
    img_height_pixels = image_data.shape[0]
    img_width_pixels = image_data.shape[1]
    display_ping_range = max_ping - min_ping
    x_pixel_positions = np.arange(img_width_pixels, dtype=np.float64)
    width_denominator = max(img_width_pixels - 1, 1)
    axis_mode = str(x_axis_mode).strip().lower() if x_axis_mode is not None else "true_time"
    if axis_mode not in {"true_time", "linear"}:
        print(
            f"Warning: x_axis_mode={x_axis_mode!r} is invalid; "
            "using 'true_time' (supported: 'true_time', 'linear')."
        )
        axis_mode = "true_time"

    # Validate optional x-axis values and decide coordinate mapping mode.
    has_custom_x_axis = (
        x_axis_values is not None and
        len(x_axis_values) == img_width_pixels
    )
    if x_axis_values is not None and not has_custom_x_axis:
        print(
            "Warning: x_axis_values length does not match image width; "
            "falling back to ping-index x-axis."
        )

    if has_custom_x_axis:
        x_axis_values = np.asarray(x_axis_values, dtype=np.float64)
        if np.any(~np.isfinite(x_axis_values)) or np.any(np.diff(x_axis_values) <= 0):
            print(
                "Warning: x_axis_values must be finite and strictly increasing; "
                "falling back to ping-index x-axis."
            )
            has_custom_x_axis = False

    x_extent_start = min_ping
    x_extent_end = max_ping
    use_custom_true_time = False
    use_custom_linear = False

    if has_custom_x_axis:
        x_extent_start = float(x_axis_values[0])
        x_extent_end = float(x_axis_values[-1])
        use_custom_true_time = axis_mode == "true_time"
        use_custom_linear = axis_mode == "linear"

    if use_custom_true_time:
        # Build per-column x edges so image pixels align with true ping timestamps.
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

        y_edges = np.linspace(depth_start, depth_stop, img_height_pixels + 1, dtype=np.float64)
        im = ax.pcolormesh(
            x_edges,
            y_edges,
            image_data,
            shading='auto',
            cmap='viridis',
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_ylim(depth_stop, depth_start)
        ax.set_xlim(float(x_edges[0]), float(x_edges[-1]))
    else:
        im = ax.imshow(image_data, aspect='auto', cmap='viridis',
                       extent=[x_extent_start, x_extent_end, depth_stop, depth_start],
                       vmin=vmin, vmax=vmax)

    def map_x_coordinate(clamped_px):
        if use_custom_true_time:
            return np.interp(clamped_px, x_pixel_positions, x_axis_values)
        if use_custom_linear:
            return x_extent_start + (clamped_px / width_denominator) * (x_extent_end - x_extent_start)
        return min_ping + (clamped_px / width_denominator) * display_ping_range
    
    # Add colorbar with specific position
    cbar_ax = plt.axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.set_label('Sv (dB)')
    
    if x_axis_label is None:
        x_axis_label = 'Time (UTC)' if x_axis_is_time else 'Ping Number'
    ax.set_xlabel(x_axis_label)
    ax.set_ylabel('Depth (m)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    if x_axis_is_time:
        locator = mdates.AutoDateLocator(minticks=4, maxticks=10)
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        plt.setp(ax.get_xticklabels(), rotation=20, ha='right')
    
    # Create a legend handle list
    legend_handles = []
    
    if contour_info is None:
        # Original behavior - each contour gets its own color
        colors = get_distinct_colors(len(contours))
        
        for idx, contour in enumerate(contours):
            points = contour.reshape(-1, 2)
            scaled_x = np.zeros(len(points))
            scaled_y = np.zeros(len(points))
            
            for i, (px, py) in enumerate(points):
                clamped_px = np.clip(px, 0, img_width_pixels - 1)
                scaled_x[i] = map_x_coordinate(clamped_px)
                depth_fraction = py / img_height_pixels
                scaled_y[i] = depth_start + depth_fraction * (depth_stop - depth_start)
            
            color = colors[idx]
            fill = ax.fill(scaled_x, scaled_y, color=color, alpha=0.3)
            
            if show_outline:
                ax.plot(scaled_x, scaled_y, color=color, linewidth=1.0)
            
            legend_handles.append(plt.Rectangle((0,0), 1, 1, fc=color, alpha=0.3, 
                                             ec=color if show_outline else "none",
                                             label=f'Layer {idx+1}'))
    else:
        # Combined view - use provided colors for each group
        for contour_list, colors, label_prefix in contour_info:
            for idx, (contour, color) in enumerate(zip(contour_list, colors)):
                points = contour.reshape(-1, 2)
                scaled_x = np.zeros(len(points))
                scaled_y = np.zeros(len(points))
                
                for i, (px, py) in enumerate(points):
                    clamped_px = np.clip(px, 0, img_width_pixels - 1)
                    scaled_x[i] = map_x_coordinate(clamped_px)
                    depth_fraction = py / img_height_pixels
                    scaled_y[i] = depth_start + depth_fraction * (depth_stop - depth_start)
                
                fill = ax.fill(scaled_x, scaled_y, color=color, alpha=0.3)
                
                if show_outline:
                    ax.plot(scaled_x, scaled_y, color=color, linewidth=1.0)
                
                legend_handles.append(plt.Rectangle((0,0), 1, 1, fc=color, alpha=0.3, 
                                                 ec=color if show_outline else "none",
                                                 label=f'{label_prefix} Layer {idx+1}'))
    
    # Add legend below the plot
    ax.legend(handles=legend_handles, 
             loc='upper center',
             bbox_to_anchor=(0.5, -0.15),
             ncol=min(5, len(legend_handles)),  # Show up to 5 items per row
             title="DVM Layers")
    
    # Adjust layout to make room for legend at bottom
    plt.subplots_adjust(right=0.9, bottom=0.2)
    
    if save_path:
        if not os.path.exists(figures_dir):
            os.makedirs(figures_dir)
        full_save_path = os.path.join(figures_dir, save_path)
        plt.savefig(full_save_path, bbox_inches='tight', dpi=300)
        print(f"Saved echogram with DVM to: {full_save_path}")
    
    plt.close()

def export_dsl_layers_to_csv(contours, original_echogram_df, min_ping, max_ping, 
                            depth_start, depth_stop, image_shape, output_dir, 
                            layer_prefix="main_dsl"):
    """
    Export detected DSL contours as separate Sv CSV files that can be imported back into Echoview.
    Each contour becomes a separate CSV file with the original data structure but only 
    containing Sv values within the detected layer boundaries.
    
    Args:
        contours: List of detected contours
        original_echogram_df: Original echogram DataFrame with all columns
        min_ping: Minimum ping number used in detection
        max_ping: Maximum ping number used in detection  
        depth_start: Starting depth used in detection
        depth_stop: Stopping depth used in detection
        image_shape: Shape of the cropped image used for detection (height, width)
        output_dir: Directory to save CSV files
        layer_prefix: Prefix for output filenames
    """
    print(f"\nStarting CSV export for {len(contours)} {layer_prefix} layers...")
    
    if not contours:
        print("No contours to export.")
        return
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get the cropped echogram data that matches our detection range
    ping_mask = (original_echogram_df['Ping_index'] >= min_ping) & \
                (original_echogram_df['Ping_index'] <= max_ping)
    cropped_echogram = original_echogram_df[ping_mask].copy()
    
    # Get depth parameters
    original_depth_start = cropped_echogram.iloc[0]['Depth_start']
    original_depth_stop = cropped_echogram.iloc[0]['Depth_stop']
    original_total_depth = original_depth_stop - original_depth_start
    sample_count = int(cropped_echogram.iloc[0]['Sample_count'])
    
    # Calculate depth indices for the region we analyzed
    pixels_per_meter = image_shape[0] / (depth_stop - depth_start)
    
    # Calculate sample indices that correspond to our analyzed depth range
    samples_per_meter = sample_count / original_total_depth
    depth_sample_start = int((depth_start - original_depth_start) * samples_per_meter)
    depth_sample_end = int((depth_stop - original_depth_start) * samples_per_meter)
    
    # Ensure indices are within bounds
    depth_sample_start = max(0, depth_sample_start)
    depth_sample_end = min(sample_count, depth_sample_end)
    
    print(f"Original depth range: {original_depth_start:.1f}m to {original_depth_stop:.1f}m")
    print(f"Analysis depth range: {depth_start:.1f}m to {depth_stop:.1f}m")
    print(f"Sample indices: {depth_sample_start} to {depth_sample_end}")
    
    # Get sample column names
    sample_columns = [col for col in cropped_echogram.columns 
                     if col.startswith('Sample_') and col != 'Sample_count']
    
    # Process each contour
    for idx, contour in enumerate(contours):
        print(f"\nProcessing {layer_prefix} layer {idx+1}...")
        
        # Create a copy of the cropped echogram for this layer
        layer_echogram = cropped_echogram.copy()
        
        # Create a mask for this contour
        contour_mask = np.zeros(image_shape, dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 1, thickness=cv2.FILLED)
        
        # Initialize all sample values to 0.0 (indicating no data/background)
        background_val = 0.0
        
        # Set all sample values to 0.0 initially
        for col in sample_columns:
            layer_echogram[col] = background_val
        
        # Only keep values within the contour - improved mapping to avoid gaps
        # Process all pings in the echogram, not just discrete pixel positions
        for _, ping_row in layer_echogram.iterrows():
            actual_ping = ping_row['Ping_index']
            ping_row_idx = ping_row.name
            
            # Convert ping number to image x-coordinate
            ping_x_coord = (actual_ping - min_ping) / (max_ping - min_ping) * image_shape[1]
            ping_x_coord = max(0, min(image_shape[1] - 1, ping_x_coord))
            
            # Check all depth samples for this ping
            for sample_idx in range(depth_sample_start, depth_sample_end):
                # Convert sample index to image y-coordinate
                depth_y_coord = (sample_idx - depth_sample_start) / (depth_sample_end - depth_sample_start) * image_shape[0]
                depth_y_coord = max(0, min(image_shape[0] - 1, depth_y_coord))
                
                # Check if this position is within the contour using bilinear interpolation
                x_floor, x_ceil = int(ping_x_coord), min(int(ping_x_coord) + 1, image_shape[1] - 1)
                y_floor, y_ceil = int(depth_y_coord), min(int(depth_y_coord) + 1, image_shape[0] - 1)
                
                # Sample the mask at the four nearest pixels and interpolate
                mask_samples = [
                    contour_mask[y_floor, x_floor],
                    contour_mask[y_floor, x_ceil],
                    contour_mask[y_ceil, x_floor],
                    contour_mask[y_ceil, x_ceil]
                ]
                
                # If any of the nearby mask pixels are set, include this sample
                if any(mask_samples):
                    sample_col = sample_columns[sample_idx]
                    # Copy the original value from the source data
                    original_value = cropped_echogram.loc[ping_row_idx, sample_col]
                    layer_echogram.loc[ping_row_idx, sample_col] = original_value
        
        # Save the layer CSV
        layer_filename = f"{layer_prefix}_layer_{idx+1}.sv.csv"
        layer_path = os.path.join(output_dir, layer_filename)
        
        # Save with the same format as the original (no sample column headers)
        # First, get the fixed column names (first 13 columns)
        fixed_cols = ['Ping_index', 'Distance_gps', 'Distance_vl', 'Ping_date',
                     'Ping_time', 'Ping_milliseconds', 'Latitude', 'Longitude', 
                     'Depth_start', 'Depth_stop', 'Range_start', 'Range_stop', 'Sample_count']
        
        # Create a custom header that only includes the fixed column names
        # The sample columns will have no headers (empty strings)
        all_columns = fixed_cols + [''] * len(sample_columns)
        
        # Save the CSV with custom header
        with open(layer_path, 'w') as f:
            # Write the header line (only fixed columns have names)
            f.write(','.join(all_columns) + '\n')
            
            # Write the data rows
            for _, row in layer_echogram.iterrows():
                row_data = []
                # Add fixed column values
                for col in fixed_cols:
                    row_data.append(str(row[col]))
                # Add sample column values
                for col in sample_columns:
                    row_data.append(str(row[col]))
                f.write(','.join(row_data) + '\n')
        
        # Count non-zero values to report (actual Sv data within the layer)
        non_zero_count = 0
        for col in sample_columns:
            non_zero_count += (layer_echogram[col] != background_val).sum()
        
        print(f"Saved {layer_filename} with {non_zero_count} Sv data points")
        print(f"Layer {idx+1} covers ping range: {layer_echogram['Ping_index'].min()} to {layer_echogram['Ping_index'].max()}")
    
    print(f"\nCSV export complete! Files saved to: {output_dir}")

def export_dsl_layers_to_boolean_csv(contours, original_echogram_df, min_ping, max_ping, 
                                    depth_start, depth_stop, image_shape, output_dir, 
                                    layer_prefix="main_dsl"):
    """
    Export detected DSL contours as boolean CSV files for Echoview.
    Each contour becomes a separate CSV file with 1s inside the detected layer boundaries
    and 0s everywhere else.
    
    Args:
        contours: List of detected contours
        original_echogram_df: Original echogram DataFrame with all columns
        min_ping: Minimum ping number used in detection
        max_ping: Maximum ping number used in detection  
        depth_start: Starting depth used in detection
        depth_stop: Stopping depth used in detection
        image_shape: Shape of the cropped image used for detection (height, width)
        output_dir: Directory to save CSV files
        layer_prefix: Prefix for output filenames
    """
    print(f"\nStarting boolean CSV export for {len(contours)} {layer_prefix} layers...")
    
    if not contours:
        print("No contours to export.")
        return
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get the cropped echogram data that matches our detection range
    ping_mask = (original_echogram_df['Ping_index'] >= min_ping) & \
                (original_echogram_df['Ping_index'] <= max_ping)
    cropped_echogram = original_echogram_df[ping_mask].copy()
    
    # Get depth parameters
    original_depth_start = cropped_echogram.iloc[0]['Depth_start']
    original_depth_stop = cropped_echogram.iloc[0]['Depth_stop']
    original_total_depth = original_depth_stop - original_depth_start
    sample_count = int(cropped_echogram.iloc[0]['Sample_count'])
    
    # Calculate sample indices that correspond to our analyzed depth range
    samples_per_meter = sample_count / original_total_depth
    depth_sample_start = int((depth_start - original_depth_start) * samples_per_meter)
    depth_sample_end = int((depth_stop - original_depth_start) * samples_per_meter)
    
    # Ensure indices are within bounds
    depth_sample_start = max(0, depth_sample_start)
    depth_sample_end = min(sample_count, depth_sample_end)
    
    print(f"Original depth range: {original_depth_start:.1f}m to {original_depth_stop:.1f}m")
    print(f"Analysis depth range: {depth_start:.1f}m to {depth_stop:.1f}m")
    print(f"Sample indices: {depth_sample_start} to {depth_sample_end}")
    
    # Get sample column names
    sample_columns = [col for col in cropped_echogram.columns 
                     if col.startswith('Sample_') and col != 'Sample_count']
    
    # Precompute ping/depth mapping arrays once (same for every contour in this export call).
    ping_values = cropped_echogram['Ping_index'].to_numpy(dtype=np.float32)
    ping_count = ping_values.shape[0]
    image_height, image_width = image_shape

    # Avoid divide-by-zero when a range collapses to a single ping/sample.
    ping_span = max_ping - min_ping
    depth_span_samples = depth_sample_end - depth_sample_start

    # Prepare x-coordinate lookup from ping values -> image pixels.
    if ping_span == 0:
        ping_x_coord = np.zeros(ping_count, dtype=np.float32)
    else:
        ping_x_coord = ((ping_values - min_ping) / ping_span) * image_width
    ping_x_coord = np.clip(ping_x_coord, 0, image_width - 1)
    x_floor = ping_x_coord.astype(np.int32)
    x_ceil = np.minimum(x_floor + 1, image_width - 1)

    # Prepare y-coordinate lookup from sample indices -> image pixels.
    sample_indices = np.arange(depth_sample_start, depth_sample_end, dtype=np.int32)
    if depth_span_samples <= 0:
        depth_y_coord = np.array([], dtype=np.float32)
    else:
        depth_y_coord = ((sample_indices - depth_sample_start) / depth_span_samples) * image_height
    depth_y_coord = np.clip(depth_y_coord, 0, image_height - 1)
    y_floor = depth_y_coord.astype(np.int32)
    y_ceil = np.minimum(y_floor + 1, image_height - 1)

    # Process each contour
    for idx, contour in enumerate(contours):
        print(f"\nProcessing boolean {layer_prefix} layer {idx+1}...")
        
        # Create a copy of the cropped echogram for this layer
        layer_echogram = cropped_echogram.copy()
        
        # Create a mask for this contour
        contour_mask = np.zeros(image_shape, dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 1, thickness=cv2.FILLED)
        
        # Build the sample grid as a dense NumPy array, then assign once to pandas.
        layer_sample_matrix = np.zeros((ping_count, len(sample_columns)), dtype=np.uint8)

        if sample_indices.size > 0 and ping_count > 0:
            # Match prior behavior: include a point if any of 4 neighboring mask pixels are set.
            mask00 = contour_mask[y_floor[:, None], x_floor[None, :]]
            mask01 = contour_mask[y_floor[:, None], x_ceil[None, :]]
            mask10 = contour_mask[y_ceil[:, None], x_floor[None, :]]
            mask11 = contour_mask[y_ceil[:, None], x_ceil[None, :]]
            inside_mask = (mask00 | mask01 | mask10 | mask11) > 0  # shape: depth x ping

            # Convert to ping-major layout and place into the corresponding sample columns.
            layer_sample_matrix[:, sample_indices] = inside_mask.T.astype(np.uint8)

        # Assign all sample columns in one shot (much faster than cell-by-cell .loc writes).
        layer_echogram.loc[:, sample_columns] = layer_sample_matrix
        
        # Save the boolean layer CSV
        layer_filename = f"{layer_prefix}_layer_{idx+1}.boolean.csv"
        layer_path = os.path.join(output_dir, layer_filename)
        
        # Save with the same format as the original (no sample column headers)
        # First, get the fixed column names (first 13 columns)
        fixed_cols = ['Ping_index', 'Distance_gps', 'Distance_vl', 'Ping_date',
                     'Ping_time', 'Ping_milliseconds', 'Latitude', 'Longitude', 
                     'Depth_start', 'Depth_stop', 'Range_start', 'Range_stop', 'Sample_count']
        
        # Create a custom header that only includes the fixed column names
        # The sample columns will have no headers (empty strings)
        all_columns = fixed_cols + [''] * len(sample_columns)
        
        # Save the CSV with custom header
        with open(layer_path, 'w') as f:
            # Write the header line (only fixed columns have names)
            f.write(','.join(all_columns) + '\n')
            
            # Write the data rows
            for _, row in layer_echogram.iterrows():
                row_data = []
                # Add fixed column values
                for col in fixed_cols:
                    row_data.append(str(row[col]))
                # Add sample column values (0 or 1)
                for col in sample_columns:
                    row_data.append(str(int(row[col])))  # Ensure integer format
                f.write(','.join(row_data) + '\n')
        
        # Count the number of 1s to report
        ones_count = int(layer_sample_matrix.sum())
        
        print(f"Saved {layer_filename} with {ones_count} layer boundary points (1s)")
        print(f"Layer {idx+1} covers ping range: {layer_echogram['Ping_index'].min()} to {layer_echogram['Ping_index'].max()}")
    
    print(f"\nBoolean CSV export complete! Files saved to: {output_dir}")

def enhance_resampled_data(sv_data, dsl_sv_threshold_min, dsl_sv_threshold_max, enhance_contrast=True, sharpen_edges=True):
    """
    Enhance resampled echogram data to improve DSL detection accuracy.
    
    Args:
        sv_data: Original Sv data array (resampled/averaged)
        dsl_sv_threshold_min: Minimum Sv value for thresholding
        dsl_sv_threshold_max: Maximum Sv value for thresholding
        enhance_contrast: Apply contrast enhancement
        sharpen_edges: Apply edge sharpening to recover blurred boundaries
        
    Returns:
        enhanced_sv_data: Enhanced Sv data array
    """
    enhanced_data = sv_data.copy()
    
    if enhance_contrast:
        # Apply histogram equalization to enhance contrast
        # Convert to 8-bit for CLAHE
        sv_8bit = preprocess_for_opencv(enhanced_data, dsl_sv_threshold_min, dsl_sv_threshold_max)
        
        # Apply Contrast Limited Adaptive Histogram Equalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced_8bit = clahe.apply(sv_8bit)
        
        # Convert back to Sv scale approximately
        min_sv = np.min(enhanced_data)
        max_sv = np.max(enhanced_data)
        enhanced_data = (enhanced_8bit.astype(np.float32) / 255.0) * (max_sv - min_sv) + min_sv
    
    if sharpen_edges and HAS_SCIPY:
        # Apply unsharp masking to sharpen blurred edges
        # Create a Gaussian blur
        blurred = ndimage.gaussian_filter(enhanced_data, sigma=1.0)
        
        # Subtract blurred from original to get high-frequency components
        high_freq = enhanced_data - blurred
        
        # Add back high-frequency components with amplification
        sharpening_factor = 1.5
        enhanced_data = enhanced_data + sharpening_factor * high_freq
        
        # Ensure values stay within reasonable Sv range
        enhanced_data = np.clip(enhanced_data, -90, -30)
    
    return enhanced_data

# This function is no longer used - the main processing is now in echogram_processing.py

if __name__ == '__main__':
    # This section can be used for testing the DSL tracking functions
    pass 