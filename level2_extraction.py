# level2_extraction.py
import cv2
import pandas as pd
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

    separator_chars = {'-', '_'}

    separator_count = sum(1 for char in text if char in separator_chars)

    if separator_count >= 2 and (separator_count / len(text)) >= 0.8:
        return True

    return False

def _extract_tag_from_ocr_results_level2(ocr_results_for_roi, text_concat_separator, ocr_y_tolerance, debug_mode):
    """
    Extracts a tag from OCR results with very lenient rules, suitable for Level 2 processing.
    Groups text fragments into lines and joins them. No strict character count or line count validation.
    Separator lines are still skipped.
    Returns the formatted tag string or None if no meaningful text is found.
    """
    if not ocr_results_for_roi:
        return None

    text_fragments_with_coords = []
    for item in ocr_results_for_roi:
        text = str(item.get('Text', '')).strip()
        bbox = item.get('BoundingBox')

        if not isinstance(bbox, list) or len(bbox) < 1:
            continue

        try:
            x_left = bbox[0][0]
            y_top = bbox[0][1]
            # Only include non-empty text fragments for grouping
            if text:
                text_fragments_with_coords.append({'text': text, 'y': y_top, 'x': x_left})
        except (IndexError, TypeError):
            print(f"Error parsing bounding box in L2 extraction: {item}")
            continue

    if not text_fragments_with_coords:
        return None

    # Sort fragments primarily by Y-coordinate (for lines) then by X-coordinate (for words within a line)
    text_fragments_with_coords.sort(key=lambda item: (item['y'], item['x']))

    current_line_y = -1
    current_line_fragments = []
    lines_of_text = []

    # Group fragments into lines based on Y-coordinate proximity
    for fragment in text_fragments_with_coords:
        if not fragment['text']:
            continue

        if current_line_y == -1 or abs(fragment['y'] - current_line_y) > ocr_y_tolerance:
            if current_line_fragments:
                lines_of_text.append(sorted(current_line_fragments, key=lambda item: item['x']))
            current_line_fragments = [fragment]
            current_line_y = fragment['y']
        else:
            current_line_fragments.append(fragment)

    # Add the last accumulated line
    if current_line_fragments:
        lines_of_text.append(sorted(current_line_fragments, key=lambda item: item['x']))

    if not lines_of_text:
        return None

    extracted_lines = []
    for line_data in lines_of_text:
        line_text = "".join([f['text'] for f in line_data if f['text']]).strip()

        # Still filter out separator lines even in lenient mode, as they are not part of the tag
        if _is_separator_line(line_text):
            if debug_mode:
                print(f"DEBUG: Skipping detected separator line in L2 extraction: '{line_text}'")
            continue

        if line_text:
            extracted_lines.append(line_text)

    if not extracted_lines:
        return None

    return text_concat_separator.join(extracted_lines)


def _detect_rectangles(image):
    """
    Detects square-like rectangles in the image.
    This is used to identify potential "System" instrument frames.
    Returns a list of dictionaries with 'center_x', 'center_y', 'width', 'height', 'bbox' for each rectangle.
    """
    rectangles = []

    # Canny edge detection with adjusted thresholds for better rectangle detection
    edges = cv2.Canny(image, 30, 90)

    # Find contours in the edged image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Approximate the contour to a polygon
        perimeter = cv2.arcLength(contour, True)
        # Slightly increased epsilon to allow for slightly less perfect quadrilaterals
        approx = cv2.approxPolyDP(contour, 0.05 * perimeter, True)

        # A quadrilateral (square or rectangle) has 4 vertices
        if len(approx) == 4:
            # Get bounding box of the approximated polygon
            x, y, w, h = cv2.boundingRect(approx)

            # Filter based on aspect ratio to find square-like shapes
            aspect_ratio = float(w) / h
            # Loosen aspect ratio range slightly for squares (was 0.85-1.15)
            if 0.7 <= aspect_ratio <= 1.3:
                # Also filter by a minimum size to avoid noise
                if w > 10 and h > 10: # Slightly reduced minimum side length (was 15)
                    # Calculate center
                    center_x = x + w / 2
                    center_y = y + h / 2
                    rectangles.append({
                        'center_x': center_x,
                        'center_y': center_y,
                        'width': w,
                        'height': h,
                        'bbox': (x, y, x + w, y + h) # Store (x1, y1, x2, y2)
                    })
    return rectangles

def extract_instruments(pil_image, blurred_image, vision_client, dynamic_min_radius, dynamic_max_radius, legend_df, legend_types, filename_base, config_params, debug_params, font):
    """
    Performs Level 2 extraction to detect and extract instrument details.
    It uses the calibrated radius range to find all circles, performs OCR,
    determines location (Field/System), and matches with legend data.
    """
    debug_mode = debug_params['DEBUG_MODE']
    debug_output_folder = debug_params['DEBUG_OUTPUT_FOLDER']
    text_concat_separator = config_params['TEXT_CONCAT_SEPARATOR']
    ocr_roi_margin_factor = config_params['OCR_ROI_MARGIN_FACTOR']
    ocr_y_tolerance = config_params['OCR_Y_TOLERANCE']


    # Prepare for debug drawing on a copy of the PIL image if in debug mode
    if debug_mode and pil_image:
        draw = ImageDraw.Draw(pil_image) # Use the passed pil_image (which is a copy) for drawing

    print(f"Starting Level 2: Detecting circles on full image using radius range {dynamic_min_radius}-{dynamic_max_radius} pixels.")
    print(f"DEBUG: Calling HoughCircles for full image with shape {blurred_image.shape}, minR={dynamic_min_radius}, maxR={dynamic_max_radius}")

    # Perform HoughCircles on the entire blurred image using the determined dynamic radius range
    circles_final = cv2.HoughCircles(
        blurred_image,
        cv2.HOUGH_GRADIENT,
        dp=config_params['HOUGH_DP'],
        minDist=config_params['HOUGH_MIN_DIST'],
        param1=config_params['HOUGH_PARAM1'],
        param2=config_params['HOUGH_PARAM2'],
        minRadius=dynamic_min_radius,  # Use dynamic/calibrated radius
        maxRadius=dynamic_max_radius   # Use dynamic/calibrated radius
    )
    print(f"DEBUG: HoughCircles call for full image returned. Result: {circles_final is not None}")

    # If no circles are detected in Level 2, return empty DataFrame
    if circles_final is None:
        print(f"No instruments detected in Level 2 for {filename_base}.pdf with dynamic radius {dynamic_min_radius}-{dynamic_max_radius}.")
        return pd.DataFrame()

    circles_final = np.uint16(np.around(circles_final)) # Convert circle parameters to integers
    print(f"Level 2: Detected {len(circles_final[0])} potential circles for final processing.")

    # Detect potential square/rectangles on the full image for 'Location' determination
    potential_squares = _detect_rectangles(blurred_image)
    if debug_mode and pil_image:
        print(f"Detected {len(potential_squares)} potential squares in the image for location analysis.")
        # Draw all detected squares in a distinct color for general debugging
        for sq in potential_squares:
            x1, y1, x2, y2 = sq['bbox']
            draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 255), width=2) # Magenta for all detected squares


    raw_ocr_data_final_circles = [] # To store all raw OCR results for debug
    circle_details_map_final = {} # To store parsed details for each circle

    # Iterate through circles detected in Level 2 for OCR and text extraction
    for i, c in enumerate(circles_final[0]):
        center_x, center_y, radius = c[0], c[1], c[2]

        # Define OCR ROI with margin around the circle
        margin = int(radius * ocr_roi_margin_factor)
        x1 = max(0, center_x - margin)
        y1 = max(0, center_y - margin)
        x2 = min(pil_image.width, center_x + margin)
        y2 = min(pil_image.height, center_y + margin)

        roi_image = pil_image.crop((x1, y1, x2, y2))

        # Perform OCR on the ROI
        img_byte_arr = io.BytesIO()
        roi_image.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()

        image = vision.Image(content=img_byte_arr)
        response = vision_client.text_detection(image=image)
        texts = response.text_annotations

        current_circle_ocr_results = []
        if texts:
            for text_annotation in texts[1:]: # Again, skip first full text annotation
                bbox_coords_list = [(v.x, v.y) for v in text_annotation.bounding_poly.vertices]
                raw_ocr_data_final_circles.append({
                    'Circle_ID': i,
                    'Circle_X': center_x,
                    'Circle_Y': center_y,
                    'Circle_Radius': radius,
                    'Text': text_annotation.description,
                    'BoundingBox': bbox_coords_list
                })
                current_circle_ocr_results.append({
                    'Text': text_annotation.description,
                    'BoundingBox': bbox_coords_list
                })

        # Store OCR results with circle details
        circle_details_map_final[i] = {
            'center_x': center_x,
            'center_y': center_y,
            'radius': radius,
            'ocr_results': current_circle_ocr_results,
            'validated_tag': None,
            'instrument_type': None,
            'location': 'Field' # Default location
        }

    # Save raw OCR output for all Level 2 circles if debug mode is on
    if debug_mode and raw_ocr_data_final_circles:
        raw_ocr_df_final = pd.DataFrame(raw_ocr_data_final_circles)
        raw_ocr_csv_path_final = os.path.join(debug_output_folder, f"{filename_base}_final_ocr_output.csv")
        raw_ocr_df_final.to_csv(raw_ocr_csv_path_final, index=False)
        print(f"Saved final raw OCR output to: {raw_ocr_csv_path_final}")


    final_instruments_data = []

    # Post-process each detected circle using lenient L2 text extraction
    for circle_id, details in circle_details_map_final.items():
        center_x, center_y, radius = details['center_x'], details['center_y'], details['radius']

        # Use the lenient Level 2 tag extraction
        extracted_tag = _extract_tag_from_ocr_results_level2(
            details['ocr_results'], text_concat_separator, ocr_y_tolerance, debug_mode
        )
        details['validated_tag'] = extracted_tag

        # If no meaningful tag could be extracted (even with lenient rules), skip this circle
        if extracted_tag is None or extracted_tag.strip() == "":
            if debug_mode and pil_image:
                # Draw a grey circle and "NO TEXT FOUND" message for debugging
                draw.ellipse((center_x - radius, center_y - radius, center_x + radius, center_y + radius),
                             outline=(128, 128, 128), width=5)

                display_text = "NO TEXT FOUND"
                lines_to_draw = display_text.split('\n')
                text_height_per_line = font.getbbox("Tg")[3] - font.getbbox("Tg")[1]
                text_draw_y_start = center_y + radius + 10

                for idx, line_text in enumerate(lines_to_draw):
                    if len(line_text) > 30:
                        line_text = line_text[:27] + "..."

                    text_bbox_at_origin = font.getbbox(line_text)
                    text_width = text_bbox_at_origin[2] - text_bbox_at_origin[0]
                    text_draw_x = center_x - text_width / 2
                    text_draw_y = text_draw_y_start + (idx * text_height_per_line)

                    text_draw_x = max(0, text_draw_x)
                    text_draw_y = max(0, text_draw_y)

                    draw.text((text_draw_x, text_draw_y), line_text, fill=(128, 128, 128), font=font)
            continue # Skip this circle if no tag could be extracted at all.

        instrument_type_from_tag = extracted_tag.split(text_concat_separator)[0].strip().upper()
        details['instrument_type'] = instrument_type_from_tag

        final_instrument_type = instrument_type_from_tag
        final_description = ""
        location = 'Field' # Default location is 'Field'

        # Determine Location (Field/System) by checking proximity to detected squares
        circle_diameter = 2 * radius
        size_tolerance_factor = 0.20 # Allow 20% size difference for square match
        center_proximity_tolerance = radius * 0.5 # Square center must be within 50% of circle radius from circle center

        for sq in potential_squares:
            sq_center_x = sq['center_x']
            sq_center_y = sq['center_y']
            sq_side = (sq['width'] + sq['height']) / 2 # Average side length of the square

            # Check if square size is similar to circle diameter
            if abs(sq_side - circle_diameter) < (circle_diameter * size_tolerance_factor):
                # Check if square center is close to circle center
                distance_to_center = math.sqrt((sq_center_x - center_x)**2 + (sq_center_y - center_y)**2)
                if distance_to_center < center_proximity_tolerance:
                    location = 'System' # Mark as 'System' if a matching square is found nearby
                    if debug_mode and pil_image:
                        print(f"  Found SYSTEM frame for '{extracted_tag}': Circle R={radius}, Square Side={sq_side:.2f}, Center Dist={distance_to_center:.2f}")
                        x1, y1, x2, y2 = sq['bbox']
                        draw.rectangle([x1, y1, x2, y2], outline=(255, 255, 0), width=5) # Draw yellow rectangle for system instruments
                    break # Found a system frame, no need to check other squares

        details['location'] = location

        # If legend is available, try to match instrument type and get description
        if legend_types and instrument_type_from_tag:
            legend_match = legend_df[legend_df['Instrument_Type'].str.upper() == instrument_type_from_tag]
            if not legend_match.empty:
                final_description = legend_match['Description'].iloc[0] # Get the first description if multiple matches
            else:
                print(f"Instrument '{extracted_tag}' at ({center_x},{center_y}) did not match any type in legend. (Type: {instrument_type_from_tag}). Description will be blank.")
        else:
            final_instrument_type = instrument_type_from_tag # Use the extracted type if no legend
            final_description = "" # Description remains blank if no legend

        # Add final instrument data to the list
        final_instruments_data.append({
            'P&ID_Filename': filename_base + ".pdf",
            'Instrument_Tag': extracted_tag,
            'Instrument_Type': final_instrument_type,
            'Description': final_description,
            'Location': location,
            'X_Coordinate': center_x,
            'Y_Coordinate': center_y,
            'Radius': radius
        })

        if debug_mode and pil_image: # Draw green circle for successfully processed instruments
            draw.ellipse((center_x - radius, center_y - radius, center_x + radius, center_y + radius), outline=(0, 255, 0), width=5)

            # Draw the instrument tag text on the debug image
            lines_for_drawing = extracted_tag.split(text_concat_separator) # Split tag into lines for drawing
            max_line_width = 0
            for line in lines_for_drawing:
                text_bbox_at_origin = font.getbbox(line)
                max_line_width = max(max_line_width, text_bbox_at_origin[2] - text_bbox_at_origin[0])

            text_height_per_line = font.getbbox("Tg")[3] - font.getbbox("Tg")[1] # Estimate height of a single line of text

            text_draw_x = center_x - max_line_width / 2
            text_draw_y = center_y + radius + 10 # Start drawing below the circle

            text_draw_x = max(0, text_draw_x) # Ensure text is not drawn off-canvas
            text_draw_y = max(0, text_draw_y)

            for line_idx, line_text in enumerate(lines_for_drawing):
                line_y = text_draw_y + (line_idx * text_height_per_line)
                current_line_width = font.getbbox(line_text)[2] - font.getbbox(line_text)[0]
                current_line_x = center_x - current_line_width / 2
                draw.text((current_line_x, line_y), line_text, fill=(0, 255, 0), font=font) # Draw tag in green

            # Add location text to debug image, below the tag
            location_text = f"Loc: {location}"
            loc_text_bbox = font.getbbox(location_text)
            loc_text_width = loc_text_bbox[2] - loc_text_bbox[0]
            loc_text_draw_x = center_x - loc_text_width / 2
            loc_text_draw_y = text_draw_y + (len(lines_for_drawing) * text_height_per_line) + 5
            draw.text((loc_text_draw_x, loc_text_draw_y), location_text, fill=(0, 0, 255), font=font) # Draw location in blue

            # Save debug ROI image for each processed instrument
            location_tag_for_filename = "SYSTEM" if location == "System" else "FIELD"
            roi_debug_path = os.path.join(debug_output_folder, f"{filename_base}_{location_tag_for_filename}_ROI_{circle_id+1}.jpg")
            roi_image.save(roi_debug_path, "JPEG")

    # Convert the list of instrument data to a Pandas DataFrame
    final_instruments_df = pd.DataFrame(final_instruments_data)
    return final_instruments_df

