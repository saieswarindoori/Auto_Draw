import os
import cv2
import pandas as pd
import numpy as np
from pdf2image import convert_from_bytes
from google.cloud import vision
from PIL import Image, ImageDraw, ImageFont 
import io
import math 

# Import configuration settings
from config import (
    GOOGLE_APPLICATION_CREDENTIALS_PATH, POPPLER_PATH, PDF_DPI,
    HOUGH_DP, HOUGH_MIN_DIST, HOUGH_PARAM1, HOUGH_PARAM2,
    HOUGH_MIN_RADIUS, HOUGH_MAX_RADIUS, # Now used for initial broad detection
    OCR_ROI_MARGIN_FACTOR, TEXT_CONCAT_SEPARATOR,
    DEBUG_MODE, DEBUG_OUTPUT_FOLDER,
    ANCHOR_MIN_COUNT, RADIUS_TOLERANCE_PERCENT, # Dynamic radius calibration parameters
    PID_GRID_ROWS, PID_GRID_COLS, PID_SEARCH_ORDER # New grid search parameters
)

class InstrumentProcessor:
    def __init__(self): 
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = GOOGLE_APPLICATION_CREDENTIALS_PATH
        self.vision_client = vision.ImageAnnotatorClient()
        print("Google Vision Client initialized successfully.")

        self.poppler_path = POPPLER_PATH
        print(f"InstrumentProcessor initialized with Poppler Path: {self.poppler_path if self.poppler_path else 'Not explicitly set (relying on PATH)'}")
        
        self.legend_df = pd.DataFrame() 
        self.legend_types = set()       
        print(f"Legend data initialized as empty. Please upload a legend file.")

        self.debug_mode = DEBUG_MODE
        self.debug_output_folder = DEBUG_OUTPUT_FOLDER 
        if self.debug_mode and not os.path.exists(self.debug_output_folder):
            os.makedirs(self.debug_output_folder)
        print(f"Debug mode is {'ON' if self.debug_mode else 'OFF'}. Outputs will be saved to: {self.debug_output_folder}")

    def load_legend_data(self, legend_df):
        self.legend_df = legend_df
        
        # Check for the standardized column name 'Instrument_Type'
        if 'Instrument_Type' in self.legend_df.columns:
            self.legend_types = set(self.legend_df['Instrument_Type'].astype(str).str.upper())
            # Check for 'Description' column, if not present, create it to avoid KeyError later
            if 'Description' not in self.legend_df.columns:
                self.legend_df['Description'] = '' # Add an empty description column
                print("Warning: 'Description' column not found in legend, adding an empty one.")
            
            print(f"Legend data loaded using 'Instrument_Type' and 'Description' columns.")
            print(f"Loaded {len(self.legend_types)} legend types: {sorted(list(self.legend_types))}")
        else:
            self.legend_types = set()
            self.legend_df = pd.DataFrame() # Ensure it's an empty DataFrame if required columns are missing
            print("Warning: 'Instrument_Type' column not found in legend. No types loaded for validation.")


    def _validate_and_format_tag(self, ocr_results_for_roi, max_tag_rows=3, min_chars_per_row=1, max_chars_per_row=5, y_tolerance=5):
        """
        Validates and formats an instrument tag from OCR results based on expected structure.
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
                text_fragments_with_coords.append({'text': text, 'y': y_top, 'x': x_left})
            except (IndexError, TypeError) as e:
                # print(f"Warning: Could not parse bounding box for OCR fragment: {e}")
                continue

        if not text_fragments_with_coords:
            return None

        # Sort fragments primarily by Y-coordinate (for lines) then by X-coordinate (for words within a line)
        text_fragments_with_coords.sort(key=lambda item: (item['y'], item['x']))

        current_line_y = -1
        current_line_fragments = []
        lines = [] 

        # Group fragments into lines based on Y-coordinate proximity
        for fragment in text_fragments_with_coords:
            if not fragment['text']: # Skip empty text fragments
                continue

            if current_line_y == -1 or abs(fragment['y'] - current_line_y) > y_tolerance:
                # New line detected
                if current_line_fragments:
                    lines.append(sorted(current_line_fragments, key=lambda item: item['x'])) # Add sorted fragments of previous line
                current_line_fragments = [fragment]
                current_line_y = fragment['y']
            else:
                # Same line
                current_line_fragments.append(fragment)
        
        # Add the last accumulated line
        if current_line_fragments: 
            lines.append(sorted(current_line_fragments, key=lambda item: item['x']))

        # Validate number of lines
        if len(lines) == 0 or len(lines) > max_tag_rows:
            return None 

        validated_lines_text = []
        for line in lines:
            line_text = "".join([f['text'] for f in line if f['text']]).strip()
            line_char_count = len(line_text)

            # Validate character count per line
            if not (min_chars_per_row <= line_char_count <= max_chars_per_row):
                return None 

            validated_lines_text.append(line_text)
        
        # Join validated lines with a newline character for the final tag
        return "\n".join(validated_lines_text)

    def _detect_rectangles(self, image):
        """
        Detects square-like rectangles in the image.
        Returns a list of dictionaries with 'center_x', 'center_y', 'width', 'height', 'bbox' for each rectangle.
        """
        rectangles = []
        
        # Pre-process image for contour detection
        # Canny edge detection - try more lenient thresholds
        edges = cv2.Canny(image, 30, 90) # Adjusted Canny thresholds (was 50, 150)
        
        # Find contours in the edged image
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # Approximate the contour to a polygon
            perimeter = cv2.arcLength(contour, True)
            # Slightly increased epsilon to allow for slightly less perfect quadrilaterals
            approx = cv2.approxPolyDP(contour, 0.05 * perimeter, True) # Was 0.04 * perimeter

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
                            'bbox': (x, y, x+w, y+h) # Store (x1, y1, x2, y2)
                        })
        return rectangles

    def process_pid_file(self, pid_file_path, filename_base):
        print(f"Starting processing for P&ID: {filename_base}.pdf")
        
        # Check if legend data is available for calibration
        if not self.legend_types: 
            print("Warning: No valid legend data loaded. Cannot use legend for dynamic radius calibration or type filtering. Falling back to default broad radius for all circles.")
            legend_available_for_calibration = False
        else:
            legend_available_for_calibration = True

        try:
            # Convert PDF to image
            with open(pid_file_path, 'rb') as f:
                pdf_content = f.read()

            if self.poppler_path:
                images = convert_from_bytes(pdf_content, dpi=PDF_DPI, poppler_path=self.poppler_path)
            else:
                images = convert_from_bytes(pdf_content, dpi=PDF_DPI)
            
            if not images:
                print(f"Error: No images converted from {filename_base}.pdf")
                return pd.DataFrame()

            pil_image = images[0] # Process only the first page for now
            if self.debug_mode:
                debug_img_path = os.path.join(self.debug_output_folder, f"{filename_base}_page1.jpg")
                pil_image.save(debug_img_path, "JPEG")
                print(f"Saved debug image: {debug_img_path}")

            # Convert PIL image to OpenCV format
            cv_image = np.array(pil_image)
            cv_image_rgb = cv_image[:, :, ::-1].copy() # Convert RGB to BGR for OpenCV

            # Pre-process image for circle detection
            gray_image = cv2.cvtColor(cv_image_rgb, cv2.COLOR_BGR2GRAY)
            blurred_image = cv2.medianBlur(gray_image, 5) # Apply median blur to reduce noise
            
            # Prepare for debug drawing on a copy of the PIL image
            if self.debug_mode:
                pil_image_draw = pil_image.copy()
                draw = ImageDraw.Draw(pil_image_draw)
                try:
                    font = ImageFont.truetype("arial.ttf", 30) # Adjusted font size for better visibility
                except IOError:
                    font = ImageFont.load_default() # Fallback font

            # Initialize dynamic radii with broad defaults in case no anchors are found
            dynamic_min_radius = HOUGH_MIN_RADIUS 
            dynamic_max_radius = HOUGH_MAX_RADIUS 
            found_anchor = False
            
            # Initialize list to store radii of circles that match legend types
            anchor_radii = [] 

            # --- LEVEL 1: Identify the Correct Circle (Calibration Phase) ---
            if legend_available_for_calibration:
                img_height, img_width = blurred_image.shape[:2]
                
                # Calculate grid cell dimensions
                cell_width = img_width // PID_GRID_COLS
                cell_height = img_height // PID_GRID_ROWS

                print(f"Starting Level 1: Searching for anchor circles in {len(PID_SEARCH_ORDER)} prioritized grid parts.")

                # Iterate through grid parts in the defined search order
                for part_idx_1based in PID_SEARCH_ORDER:
                    if found_anchor: # If enough anchors are found, break early
                        break

                    # Convert 1-based index to 0-based row/col for calculations
                    row_0based = (part_idx_1based - 1) // PID_GRID_COLS
                    col_0based = (part_idx_1based - 1) % PID_GRID_COLS

                    # Calculate pixel boundaries for the current grid part
                    x_start = col_0based * cell_width
                    y_start = row_0based * cell_height
                    # Ensure the last cell extends to the image edge to avoid cutting off pixels
                    x_end = (col_0based + 1) * cell_width if col_0based < PID_GRID_COLS - 1 else img_width
                    y_end = (col_0based + 1) * cell_height if row_0based < PID_GRID_ROWS - 1 else img_height

                    # Skip if ROI is invalid (e.g., due to division rounding for last cell)
                    if x_start >= x_end or y_start >= y_end:
                        # print(f"Skipping invalid grid part {part_idx_1based} (ROI dimensions invalid).")
                        continue

                    # Crop the blurred image to the current part's ROI
                    current_part_blurred = blurred_image[y_start:y_end, x_start:x_end]

                    if self.debug_mode:
                        # Draw grid lines on the debug image for visualization
                        draw.rectangle([x_start, y_start, x_end, y_end], outline=(128, 128, 128), width=3) # Grey lines
                        draw.text((x_start + 10, y_start + 10), str(part_idx_1based), fill=(128, 128, 128), font=font)


                    # Run HoughCircles on this small, cropped part using broad default radii
                    circles_in_part = cv2.HoughCircles(
                        current_part_blurred,
                        cv2.HOUGH_GRADIENT,
                        dp=HOUGH_DP,           
                        minDist=HOUGH_MIN_DIST,
                        param1=HOUGH_PARAM1,   
                        param2=HOUGH_PARAM2,   
                        minRadius=HOUGH_MIN_RADIUS,  # Use broad config default
                        maxRadius=HOUGH_MAX_RADIUS   # Use broad config default
                    )

                    if circles_in_part is not None:
                        circles_in_part = np.uint16(np.around(circles_in_part))
                        # print(f"Detected {len(circles_in_part[0])} circles in part {part_idx_1based}.")

                        for c_part in circles_in_part[0]:
                            # Convert circle coordinates from relative (part) to absolute (full image)
                            center_x_full = c_part[0] + x_start
                            center_y_full = c_part[1] + y_start
                            radius = c_part[2]

                            # Define ROI for OCR around the detected circle (on the full image)
                            margin = int(radius * OCR_ROI_MARGIN_FACTOR)
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
                            response = self.vision_client.text_detection(image=image)
                            texts = response.text_annotations

                            current_circle_ocr_results = []
                            if texts:
                                # Skip the first text annotation as it's the full text detected in the image
                                for text_annotation in texts[1:]:
                                    bbox_coords_list = [(v.x, v.y) for v in text_annotation.bounding_poly.vertices]
                                    current_circle_ocr_results.append({
                                        'Text': text_annotation.description,
                                        'BoundingBox': bbox_coords_list 
                                    })
                            
                            # Validate and format the OCR'd text into a potential instrument tag
                            validated_tag = self._validate_and_format_tag(current_circle_ocr_results)

                            if validated_tag is not None and validated_tag.strip() != "":
                                # Extract instrument type (first line of the tag) and check against legend
                                instrument_type_from_tag = validated_tag.split('\n')[0].strip().upper()
                                if instrument_type_from_tag in self.legend_types:
                                    anchor_radii.append(radius) # Add radius to our anchor candidates
                                    print(f"Found anchor circle in part {part_idx_1based}: {validated_tag} at ({center_x_full},{center_y_full}) with radius {radius}.")
                                    if self.debug_mode:
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
                                    if len(anchor_radii) >= ANCHOR_MIN_COUNT:
                                        found_anchor = True
                                        break # Break out of inner loop (circles in current part)
                    if found_anchor: # Break out of outer loop (grid parts) if enough anchors found
                        break


                # Determine dynamic radius range based on the collected anchor radii
                if len(anchor_radii) >= ANCHOR_MIN_COUNT:
                    avg_radius = np.mean(anchor_radii)
                    radius_tolerance_pixels = avg_radius * RADIUS_TOLERANCE_PERCENT
                    dynamic_min_radius = max(1, int(avg_radius - radius_tolerance_pixels)) # Ensure min_radius is at least 1
                    dynamic_max_radius = int(avg_radius + radius_tolerance_pixels)
                    print(f"Level 1 complete. Dynamic radius range determined from {len(anchor_radii)} anchors: {dynamic_min_radius}-{dynamic_max_radius} pixels (Avg: {avg_radius:.2f})")
                else:
                    # If not enough anchors were found, fall back to the broad default radii
                    print(f"Level 1 complete. Fewer than {ANCHOR_MIN_COUNT} anchor circles found. Using default broad radius range: {HOUGH_MIN_RADIUS}-{HOUGH_MAX_RADIUS} pixels for Level 2.")
            else:
                # If no legend was uploaded, Level 1 (calibration) is skipped
                print("Legend not available for calibration. Proceeding to Level 2 with default broad radius range.")

            # --- LEVEL 2: List the Instruments (Full Detection Phase) ---
            print(f"Starting Level 2: Detecting circles on full image using radius range {dynamic_min_radius}-{dynamic_max_radius} pixels.")
            
            # Perform HoughCircles on the entire blurred image using the determined dynamic radius range
            circles_final = cv2.HoughCircles(
                blurred_image,
                cv2.HOUGH_GRADIENT,
                dp=HOUGH_DP,           
                minDist=HOUGH_MIN_DIST,
                param1=HOUGH_PARAM1,   
                param2=HOUGH_PARAM2,   
                minRadius=dynamic_min_radius,  # Use dynamic/calibrated radius
                maxRadius=dynamic_max_radius   # Use dynamic/calibrated radius
            )
            
            # If no circles are detected in Level 2, return empty DataFrame
            if circles_final is None:
                print(f"No instruments detected in Level 2 for {filename_base}.pdf with dynamic radius {dynamic_min_radius}-{dynamic_max_radius}.")
                return pd.DataFrame() 

            circles_final = np.uint16(np.around(circles_final)) # Convert circle parameters to integers
            print(f"Level 2: Detected {len(circles_final[0])} potential circles for final processing.")

            # New: Detect potential square/rectangles on the full image for 'Location' determination
            potential_squares = self._detect_rectangles(blurred_image)
            if self.debug_mode:
                print(f"Detected {len(potential_squares)} potential squares in the image for location analysis.")
                # Draw all detected squares in a distinct color for general debugging
                for sq in potential_squares:
                    x1, y1, x2, y2 = sq['bbox']
                    draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 255), width=2) # Magenta for all detected squares


            raw_ocr_data_final_circles = [] # To store all raw OCR results for debug
            circle_details_map_final = {} # To store parsed details for each circle

            # Iterate through circles detected in Level 2 for OCR and validation
            for i, c in enumerate(circles_final[0]):
                center_x, center_y, radius = c[0], c[1], c[2]
                
                # Define OCR ROI with margin around the circle
                margin = int(radius * OCR_ROI_MARGIN_FACTOR)
                x1 = max(0, center_x - margin)
                y1 = max(0, center_y - margin)
                x2 = min(pil_image.width, center_x + margin)
                y2 = min(pil_image.height, center_y + margin)
                
                roi_image = pil_image.crop((x1, y1, x2, y2))

                if self.debug_mode:
                    # Save each OCR ROI for debugging
                    roi_debug_path = os.path.join(self.debug_output_folder, f"{filename_base}_FinalROI_{i+1}.jpg")
                    roi_image.save(roi_debug_path, "JPEG")

                # Perform OCR on the ROI
                img_byte_arr = io.BytesIO()
                roi_image.save(img_byte_arr, format='JPEG')
                img_byte_arr = img_byte_arr.getvalue()
                
                image = vision.Image(content=img_byte_arr)
                response = self.vision_client.text_detection(image=image)
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
            if self.debug_mode and raw_ocr_data_final_circles:
                raw_ocr_df_final = pd.DataFrame(raw_ocr_data_final_circles)
                raw_ocr_csv_path_final = os.path.join(self.debug_output_folder, f"{filename_base}_final_ocr_output.csv")
                raw_ocr_df_final.to_csv(raw_ocr_csv_path_final, index=False)
                print(f"Saved final raw OCR output to: {raw_ocr_csv_path_final}")


            final_instruments_data = []

            # Post-process and validate each detected circle
            for circle_id, details in circle_details_map_final.items():
                center_x, center_y, radius = details['center_x'], details['center_y'], details['radius']
                
                validated_tag = self._validate_and_format_tag(details['ocr_results'])
                details['validated_tag'] = validated_tag 

                # If tag validation fails, skip this circle and mark it in debug image
                if validated_tag is None or validated_tag.strip() == "":
                    if self.debug_mode: 
                        draw.ellipse((center_x - radius, center_y - radius, center_x + radius, center_y + radius), 
                                     outline=(255, 165, 0), width=5) # Orange for validation fail
                        text_to_draw = "VALIDATION FAIL"
                        text_bbox_at_origin = font.getbbox(text_to_draw)
                        text_width = text_bbox_at_origin[2] - text_bbox_at_origin[0]
                        text_draw_x = center_x - text_width / 2
                        text_draw_y = center_y + radius + 10
                        draw.text((text_draw_x, text_draw_y), text_to_draw, fill=(255, 165, 0), font=font)
                    continue

                instrument_type_from_tag = validated_tag.split('\n')[0].strip().upper()
                details['instrument_type'] = instrument_type_from_tag 

                final_instrument_type = instrument_type_from_tag
                final_description = "" # Default description to blank
                location = 'Field' # Default to 'Field' initially for each instrument

                # --- NEW LOGIC: Determine Location (Field/System) ---
                circle_diameter = 2 * radius
                # Tolerances for checking if a square matches a circle for 'System' location
                # Square side can be within +/- 20% of circle diameter
                size_tolerance_factor = 0.20 
                # Center of square must be within 50% of circle's radius from circle's center
                center_proximity_tolerance = radius * 0.5 

                for sq in potential_squares:
                    sq_center_x = sq['center_x']
                    sq_center_y = sq['center_y']
                    # Use the average of width and height for square's side approximation
                    sq_side = (sq['width'] + sq['height']) / 2 
                    
                    # Check 1: Size similarity
                    # Is the square's side length close to the circle's diameter?
                    if abs(sq_side - circle_diameter) < (circle_diameter * size_tolerance_factor):
                        # Check 2: Center proximity
                        # Is the square's center close to the circle's center?
                        distance_to_center = math.sqrt((sq_center_x - center_x)**2 + (sq_center_y - center_y)**2)
                        if distance_to_center < center_proximity_tolerance:
                            location = 'System'
                            if self.debug_mode:
                                print(f"  Found SYSTEM frame for '{validated_tag}': Circle R={radius}, Square Side={sq_side:.2f}, Center Dist={distance_to_center:.2f}")
                                # Draw a thick yellow rectangle around the system instrument's frame for debug
                                x1, y1, x2, y2 = sq['bbox']
                                draw.rectangle([x1, y1, x2, y2], outline=(255, 255, 0), width=5) # Yellow for System frame
                            break # Found a match, no need to check other squares for this circle

                details['location'] = location # Update location in details map


                # If legend is available, try to match and get description
                if legend_available_for_calibration and instrument_type_from_tag: 
                    legend_match = self.legend_df[self.legend_df['Instrument_Type'].str.upper() == instrument_type_from_tag]
                    if not legend_match.empty:
                        final_description = legend_match['Description'].iloc[0] # Get description from legend
                    else:
                        print(f"Instrument '{validated_tag}' at ({center_x},{center_y}) did not match any type in legend. (Type: {instrument_type_from_tag}). Description will be blank.")
                else:
                    # If no legend, instrument_type is just what OCR gives, description is blank
                    final_instrument_type = instrument_type_from_tag 
                    final_description = "" 

                # Add final instrument data
                final_instruments_data.append({
                    'P&ID_Filename': filename_base + ".pdf",
                    'Instrument_Tag': validated_tag,
                    'Instrument_Type': final_instrument_type,
                    'Description': final_description, 
                    'Location': location, # Add the new 'Location' column here!
                    'X_Coordinate': center_x,
                    'Y_Coordinate': center_y,
                    'Radius': radius
                })
                
                if self.debug_mode: # Draw green circle for successfully processed instruments
                    draw.ellipse((center_x - radius, center_y - radius, center_x + radius, center_y + radius), outline=(0, 255, 0), width=5) # Green for final instrument
                    
                    # Draw the instrument tag text on the debug image
                    lines_for_drawing = validated_tag.split('\n')
                    max_line_width = 0
                    for line in lines_for_drawing:
                        text_bbox_at_origin = font.getbbox(line)
                        max_line_width = max(max_line_width, text_bbox_at_origin[2] - text_bbox_at_origin[0])
                    
                    text_height_per_line = font.getbbox("Tg")[3] - font.getbbox("Tg")[1] # Approx height of one line
                    
                    text_draw_x = center_x - max_line_width / 2
                    text_draw_y = center_y + radius + 10

                    # Ensure text is drawn within image boundaries
                    text_draw_x = max(0, text_draw_x)
                    text_draw_y = max(0, text_draw_y)

                    for line_idx, line_text in enumerate(lines_for_drawing):
                        line_y = text_draw_y + (line_idx * text_height_per_line)
                        current_line_width = font.getbbox(line_text)[2] - font.getbbox(line_text)[0]
                        current_line_x = center_x - current_line_width / 2
                        draw.text((current_line_x, line_y), line_text, fill=(0, 255, 0), font=font)

                    # Add location text to debug image, below the tag
                    location_text = f"Loc: {location}"
                    loc_text_bbox = font.getbbox(location_text)
                    loc_text_width = loc_text_bbox[2] - loc_text_bbox[0]
                    loc_text_draw_x = center_x - loc_text_width / 2
                    loc_text_draw_y = text_draw_y + (len(lines_for_drawing) * text_height_per_line) + 5
                    draw.text((loc_text_draw_x, loc_text_draw_y), location_text, fill=(0, 0, 255), font=font) # Blue text for location


            final_instruments_df = pd.DataFrame(final_instruments_data)

            if self.debug_mode:
                # Save the final debug image with all detected circles and their tags
                debug_circles_img_path = os.path.join(self.debug_output_folder, f"{filename_base}_circles_detected.jpg")
                pil_image_draw.save(debug_circles_img_path, "JPEG")
                print(f"Saved debug image with detected circles: {debug_circles_img_path}")

            return final_instruments_df

        except Exception as e:
            print(f"Error processing {filename_base}.pdf: {e}")
            import traceback
            traceback.print_exc() # Print full traceback for debugging
            return pd.DataFrame()