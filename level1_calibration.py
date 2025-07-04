# level1_calibration.py
import cv2
import numpy as np
from google.cloud import vision
from PIL import Image, ImageDraw, ImageFont
import io
import os
import math

def _is_separator_line(text):
    """
    Heuristically checks if a given text string represents a separator line.
    A separator line is typically composed primarily of dashes or underscores.
    It must contain at least two separator characters.
    """
    if not text:
        return False

    # Define common separator characters
    separator_chars = {'-', '_'}

    # Count occurrences of separator characters in the text
    separator_count = sum(1 for char in text if char in separator_chars)

    # If the line consists mostly of separator characters (e.g., >80%)
    # and has at least two such characters, consider it a separator line.
    # This prevents single dashes within a tag (like "T-101") from being filtered.
    if separator_count >= 2 and (separator_count / len(text)) >= 0.8:
        return True

    return False

def _validate_and_format_tag(ocr_results_for_roi, ocr_validation_config, text_concat_separator):
    """
    Validates and formats an instrument tag from OCR results based on expected structure.
    Uses OCR validation parameters from ocr_validation_config (passed from config.py).
    This method is primarily used for Level 1 (anchor calibration) where strict validation is needed
    to ensure the detected circles are indeed instrument tags.
    Returns a tuple: (formatted_tag_string, None) on success, or (None, failure_reason_string) on failure.
    """
    if not ocr_results_for_roi:
        return (None, "No OCR results found.")

    text_fragments_with_coords = []
    # Extract text and bounding box coordinates from OCR results
    for item in ocr_results_for_roi:
        text = str(item.get('Text', '')).strip()
        bbox = item.get('BoundingBox')

        # Skip if bounding box is invalid
        if not isinstance(bbox, list) or len(bbox) < 1:
            continue

        try:
            x_left = bbox[0][0]
            y_top = bbox[0][1]
            text_fragments_with_coords.append({'text': text, 'y': y_top, 'x': x_left})
        except (IndexError, TypeError) as e:
            # Log error for malformed bounding box
            print(f"Error parsing bounding box: {e} for item: {item}")
            continue

    if not text_fragments_with_coords:
        return (None, "No valid text fragments with coordinates.")

    # Sort fragments primarily by Y-coordinate (for lines) then by X-coordinate (for words within a line)
    text_fragments_with_coords.sort(key=lambda item: (item['y'], item['x']))

    current_line_y = -1
    current_line_fragments = []
    lines = [] # List to store grouped lines of text

    # Group fragments into lines based on Y-coordinate proximity (using OCR_Y_TOLERANCE)
    for fragment in text_fragments_with_coords:
        if not fragment['text']: # Skip empty text fragments
            continue

        if current_line_y == -1 or abs(fragment['y'] - current_line_y) > ocr_validation_config['OCR_Y_TOLERANCE']:
            # New line detected: add previous line (if any) and start a new one
            if current_line_fragments:
                lines.append(sorted(current_line_fragments, key=lambda item: item['x'])) # Add sorted fragments of previous line
            current_line_fragments = [fragment]
            current_line_y = fragment['y']
        else:
            # Same line: append fragment to current line
            current_line_fragments.append(fragment)

    # Add the last accumulated line after the loop finishes
    if current_line_fragments:
        lines.append(sorted(current_line_fragments, key=lambda item: item['x']))

    # Validate number of lines before processing
    if len(lines) == 0:
        return (None, "No meaningful lines detected after Y-grouping.")

    validated_lines_text = []
    for line_idx, line in enumerate(lines):
        line_text = "".join([f['text'] for f in line if f['text']]).strip()

        # Check if this line is a separator and skip it if it is
        if _is_separator_line(line_text):
            if ocr_validation_config['DEBUG_MODE']:
                print(f"DEBUG: Skipping detected separator line: '{line_text}' for circle validation.")
            continue # Skip this line from further validation and inclusion in the tag

        line_char_count = len(line_text)

        # Validate character count per line using configurable parameters
        if not (ocr_validation_config['OCR_MIN_CHARS_PER_ROW'] <= line_char_count <= ocr_validation_config['OCR_MAX_CHARS_PER_ROW']):
            return (None, f"Line {line_idx+1} char count ({line_char_count}) outside {ocr_validation_config['OCR_MIN_CHARS_PER_ROW']}-{ocr_validation_config['OCR_MAX_CHARS_PER_ROW']} range: '{line_text}'")

        validated_lines_text.append(line_text)

    # Check the total number of meaningful lines AFTER processing all using configurable parameter
    if len(validated_lines_text) == 0 or len(validated_lines_text) > ocr_validation_config['OCR_MAX_TAG_ROWS']:
        return (None, f"Total meaningful lines ({len(validated_lines_text)}) outside 1-{ocr_validation_config['OCR_MAX_TAG_ROWS']} range.")

    # Join validated lines with the configured separator to form the final tag
    return (text_concat_separator.join(validated_lines_text), None)

def calibrate_radius_range(pil_image, blurred_image, vision_client, legend_types, config_params, debug_params, font):
    """
    Performs Level 1 calibration to determine the dynamic radius range for circle detection.
    This involves searching for circles in prioritized grid areas, performing OCR,
    and validating against the legend to find 'anchor' circles.
    """
    # Initialize dynamic radii with broad defaults in case no anchors are found
    dynamic_min_radius = config_params['HOUGH_MIN_RADIUS']
    dynamic_max_radius = config_params['HOUGH_MAX_RADIUS']
    found_anchor = False
    anchor_radii = [] # List to store radii of circles that match legend types

    legend_available_for_calibration = bool(legend_types)
    debug_mode = debug_params['DEBUG_MODE']
    debug_output_folder = debug_params['DEBUG_OUTPUT_FOLDER']

    # If no legend is available, skip calibration and return default broad range
    if not legend_available_for_calibration:
        print("Legend not available for calibration. Proceeding to Level 2 with default broad radius range.")
        return dynamic_min_radius, dynamic_max_radius

    img_height, img_width = blurred_image.shape[:2]

    # Calculate grid cell dimensions for the prioritized search
    cell_width = img_width // config_params['PID_GRID_COLS']
    cell_height = img_height // config_params['PID_GRID_ROWS']

    print(f"Starting Level 1: Searching for anchor circles in {len(config_params['PID_SEARCH_ORDER'])} prioritized grid parts.")
    print("DEBUG: About to enter Level 1 grid search loop.")

    # Prepare for debug drawing on a copy of the PIL image if in debug mode
    if debug_mode and pil_image:
        draw = ImageDraw.Draw(pil_image) # Use the passed pil_image (which is a copy) for drawing

    # Iterate through grid parts in the defined search order (from config)
    for part_idx_1based in config_params['PID_SEARCH_ORDER']:
        print(f"  Processing Grid Part {part_idx_1based}...")
        if found_anchor: # If enough anchors are found, break early
            print(f"  Enough anchors found, breaking from Level 1 search.")
            break

        # Convert 1-based index to 0-based row/col for calculations
        row_0based = (part_idx_1based - 1) // config_params['PID_GRID_COLS']
        col_0based = (part_idx_1based - 1) % config_params['PID_GRID_COLS']

        # Calculate pixel boundaries for the current grid part
        x_start = col_0based * cell_width
        y_start = row_0based * cell_height
        # Ensure the last cell extends to the image edge to avoid cutting off pixels
        x_end = (col_0based + 1) * cell_width if col_0based < config_params['PID_GRID_COLS'] - 1 else img_width
        y_end = (row_0based + 1) * cell_height if row_0based < config_params['PID_GRID_ROWS'] - 1 else img_height

        # Skip if ROI is invalid (e.g., due to division rounding for last cell)
        if x_start >= x_end or y_start >= y_end:
            print(f"  DEBUG: Skipped Grid Part {part_idx_1based} due to invalid ROI dimensions ({x_start},{y_start},{x_end},{y_end}).")
            continue

        # Crop the blurred image to the current part's ROI
        current_part_blurred = blurred_image[y_start:y_end, x_start:x_end]

        # Skip if the cropped part has zero dimensions
        if current_part_blurred.shape[0] == 0 or current_part_blurred.shape[1] == 0:
            print(f"  DEBUG: Skipped Grid Part {part_idx_1based} due to zero dimensions after slicing (ROI shape: {current_part_blurred.shape}).")
            continue

        # Draw grid lines and part number on the debug image for visualization
        if debug_mode and pil_image:
            draw.rectangle([x_start, y_start, x_end, y_end], outline=(128, 128, 128), width=3) # Grey lines
            draw.text((x_start + 10, y_start + 10), str(part_idx_1based), fill=(128, 128, 128), font=font)


        # Run HoughCircles on this small, cropped part using broad default radii
        circles_in_part = None
        print(f"    DEBUG: Calling HoughCircles for part {part_idx_1based} with ROI shape {current_part_blurred.shape}, minR={config_params['HOUGH_MIN_RADIUS']}, maxR={config_params['HOUGH_MAX_RADIUS']}")
        try:
            circles_in_part = cv2.HoughCircles(
                current_part_blurred,
                cv2.HOUGH_GRADIENT,
                dp=config_params['HOUGH_DP'],
                minDist=config_params['HOUGH_MIN_DIST'],
                param1=config_params['HOUGH_PARAM1'],
                param2=config_params['HOUGH_PARAM2'],
                minRadius=config_params['HOUGH_MIN_RADIUS'],  # Use broad config default
                maxRadius=config_params['HOUGH_MAX_RADIUS']   # Use broad config default
            )
        except Exception as hough_e:
            print(f"    ERROR: HoughCircles failed for part {part_idx_1based}: {hough_e}")
            import traceback
            traceback.print_exc()
            circles_in_part = None

        print(f"    DEBUG: HoughCircles call returned for part {part_idx_1based}. Result: {circles_in_part is not None}")


        if circles_in_part is not None:
            circles_in_part = np.uint16(np.around(circles_in_part))
            print(f"    Detected {len(circles_in_part[0])} circles in part {part_idx_1based}.")

            for c_part in circles_in_part[0]:
                # Convert circle coordinates from relative (part) to absolute (full image)
                center_x_full = c_part[0] + x_start
                center_y_full = c_part[1] + y_start
                radius = c_part[2]

                # Define ROI for OCR around the detected circle (on the full image)
                margin = int(radius * config_params['OCR_ROI_MARGIN_FACTOR'])
                roi_x1 = max(0, center_x_full - margin)
                roi_y1 = max(0, center_y_full - margin)
                roi_x2 = min(pil_image.width, center_x_full + margin)
                roi_y2 = min(pil_image.height, center_y_full + margin)

                roi_image = pil_image.crop((roi_x1, roi_y1, roi_x2, roi_y2))

                # Perform OCR on the ROI using Google Vision API
                img_byte_arr = io.BytesIO()
                roi_image.save(img_byte_arr, format='JPEG')
                img_byte_arr = img_byte_arr.getvalue()

                image = vision.Image(content=img_byte_arr)
                response = vision_client.text_detection(image=image)
                texts = response.text_annotations

                current_circle_ocr_results = []
                if texts:
                    # Skip the first text annotation as it's typically the entire text detected in the image
                    for text_annotation in texts[1:]:
                        bbox_coords_list = [(v.x, v.y) for v in text_annotation.bounding_poly.vertices]
                        current_circle_ocr_results.append({
                            'Text': text_annotation.description,
                            'BoundingBox': bbox_coords_list
                        })

                # Prepare OCR validation configuration for _validate_and_format_tag
                ocr_validation_config = {
                    'DEBUG_MODE': debug_mode,
                    'OCR_MIN_CHARS_PER_ROW': config_params['OCR_MIN_CHARS_PER_ROW'],
                    'OCR_MAX_CHARS_PER_ROW': config_params['OCR_MAX_CHARS_PER_ROW'],
                    'OCR_MAX_TAG_ROWS': config_params['OCR_MAX_TAG_ROWS'],
                    'OCR_Y_TOLERANCE': config_params['OCR_Y_TOLERANCE']
                }
                # Validate and format the OCR'd text into a potential instrument tag (STRICT VALIDATION FOR ANCHORS)
                validated_tag, validation_fail_reason = _validate_and_format_tag(
                    current_circle_ocr_results,
                    ocr_validation_config,
                    config_params['TEXT_CONCAT_SEPARATOR']
                )

                if validated_tag is not None and validated_tag.strip() != "":
                    print(f"      Validated tag: '{validated_tag}'.")
                    # Extract instrument type (first line of the tag) and check against legend
                    instrument_type_from_tag = validated_tag.split(config_params['TEXT_CONCAT_SEPARATOR'])[0].strip().upper()
                    if instrument_type_from_tag in legend_types:
                        print(f"        MATCH! Tag type '{instrument_type_from_tag}' found in legend.")
                        anchor_radii.append(radius) # Add radius to our anchor candidates
                        print(f"Found anchor circle in part {part_idx_1based}: {validated_tag} at ({center_x_full},{center_y_full}) with radius {radius}.")
                        if debug_mode and pil_image:
                            # Draw blue circle for detected anchors during debug
                            draw.ellipse((center_x_full - radius, center_y_full - radius,
                                          center_x_full + radius, center_y_full + radius),
                                         outline=(0, 0, 255), width=5) # Blue for Anchor
                            text_to_draw_anchor = f"ANCHOR_P{part_idx_1based}"
                            text_bbox_at_origin = font.getbbox(text_to_draw_anchor)
                            text_width = text_bbox_at_origin[2] - text_bbox_at_origin[0]
                            text_draw_x = center_x_full - text_width / 2
                            text_draw_y = center_y_full + radius + 5
                            draw.text((text_draw_x, text_draw_y), text_to_draw_anchor, fill=(0, 0, 255), font=font)

                        # If we found enough anchors, we can stop Level 1 early
                        if len(anchor_radii) >= config_params['ANCHOR_MIN_COUNT']:
                            found_anchor = True
                            break # Break out of inner loop (circles in current part)
                    else:
                        print(f"        NO MATCH: Tag type '{instrument_type_from_tag}' NOT found in legend.")
                else:
                    if debug_mode and pil_image: # Draw orange circle for Level 1 validation fail
                        draw.ellipse((center_x_full - radius, center_y_full - radius, center_x_full + radius, center_y_full + radius),
                                     outline=(255, 165, 0), width=5) # Orange for Level 1 validation fail
                        display_text = "L1 VALIDATION FAIL"
                        if validation_fail_reason:
                            display_text += f"\nReason: {validation_fail_reason}"
                        lines_to_draw = display_text.split('\n')
                        text_height_per_line = font.getbbox("Tg")[3] - font.getbbox("Tg")[1]
                        text_draw_y_start = center_y_full + radius + 10
                        for idx, line_text in enumerate(lines_to_draw):
                            if len(line_text) > 30:
                                line_text = line_text[:27] + "..."
                            text_bbox_at_origin = font.getbbox(line_text)
                            text_width = text_bbox_at_origin[2] - text_bbox_at_origin[0]
                            text_draw_x = center_x_full - text_width / 2
                            text_draw_y = text_draw_y_start + (idx * text_height_per_line)
                            text_draw_x = max(0, text_draw_x)
                            text_draw_y = max(0, text_draw_y)
                            draw.text((text_draw_x, text_draw_y), line_text, fill=(255, 165, 0), font=font)
                    print(f"      Validated tag is None or empty for current circle. Reason: {validation_fail_reason}")
        else:
            print(f"    No circles detected in part {part_idx_1based} (HoughCircles returned None).")

    # Determine dynamic radius range based on the collected anchor radii
    if len(anchor_radii) >= config_params['ANCHOR_MIN_COUNT']:
        avg_radius = np.mean(anchor_radii)
        radius_tolerance_pixels = avg_radius * config_params['RADIUS_TOLERANCE_PERCENT']
        dynamic_min_radius = max(1, int(avg_radius - radius_tolerance_pixels)) # Ensure min_radius is at least 1
        dynamic_max_radius = int(avg_radius + radius_tolerance_pixels)
        print(f"Level 1 complete. Dynamic radius range determined from {len(anchor_radii)} anchors: {dynamic_min_radius}-{dynamic_max_radius} pixels (Avg: {avg_radius:.2f})")
    else:
        # If not enough anchors were found, fall back to the broad default radii
        print(f"Level 1 complete. Fewer than {config_params['ANCHOR_MIN_COUNT']} anchor circles found. Using default broad radius range: {config_params['HOUGH_MIN_RADIUS']}-{config_params['HOUGH_MAX_RADIUS']} pixels for Level 2.")
        dynamic_min_radius = config_params['HOUGH_MIN_RADIUS']
        dynamic_max_radius = config_params['HOUGH_MAX_RADIUS']

    return dynamic_min_radius, dynamic_max_radius

