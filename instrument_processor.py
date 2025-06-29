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
    HOUGH_DP, HOUGH_MIN_DIST, HOUGH_PARAM1, HOUGH_PARAM2, HOUGH_MIN_RADIUS, HOUGH_MAX_RADIUS,
    OCR_ROI_MARGIN_FACTOR, TEXT_CONCAT_SEPARATOR,
    DEBUG_MODE, DEBUG_OUTPUT_FOLDER
)

class InstrumentProcessor:
    def __init__(self): 
        # Set Google Cloud credentials
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = GOOGLE_APPLICATION_CREDENTIALS_PATH
        self.vision_client = vision.ImageAnnotatorClient()
        print("Google Vision Client initialized successfully.")

        self.poppler_path = POPPLER_PATH
        print(f"InstrumentProcessor initialized with Poppler Path: {self.poppler_path if self.poppler_path else 'Not explicitly set (relying on PATH)'}")
        
        # Initialize legend_df and legend_types as empty
        self.legend_df = pd.DataFrame() # Initialize as empty DataFrame
        self.legend_types = set()       # Initialize as empty set
        print(f"Legend data initialized as empty. Please upload a legend file.")

        self.debug_mode = DEBUG_MODE
        self.debug_output_folder = DEBUG_OUTPUT_FOLDER
        if self.debug_mode and not os.path.exists(self.debug_output_folder):
            os.makedirs(self.debug_output_folder)
        print(f"Debug mode is {'ON' if self.debug_mode else 'OFF'}. Outputs will be saved to: {self.debug_output_folder}")

    def load_legend_data(self, legend_df):
        self.legend_df = legend_df
        
        # --- MODIFIED LOGIC HERE: Automatically assign column names ---
        if self.legend_df.shape[1] >= 2: # Check if there are at least two columns
            # Rename the first two columns explicitly
            self.legend_df.columns = ['Instrument Type', 'Description'] + list(self.legend_df.columns[2:])
            
            self.legend_types = set(self.legend_df['Instrument Type'].astype(str).str.upper())
            print(f"Legend data loaded using the first column as 'Instrument Type' and the second as 'Description'.")
            print(f"Loaded {len(self.legend_types)} legend types: {sorted(list(self.legend_types))}")
        elif self.legend_df.shape[1] == 1: # If only one column, treat it as Instrument Type
            self.legend_df.columns = ['Instrument Type']
            self.legend_types = set(self.legend_df['Instrument Type'].astype(str).str.upper())
            print(f"Legend data loaded using the only column as 'Instrument Type'. No 'Description' column available.")
            print(f"Loaded {len(self.legend_types)} legend types: {sorted(list(self.legend_types))}")
        else:
            self.legend_types = set()
            self.legend_df = pd.DataFrame() # Ensure legend_df is empty if no valid columns
            print("Warning: Legend file has fewer than one column. No types loaded for validation.")


    def process_pid_file(self, pid_file_path, filename_base):
        print(f"Starting processing for P&ID: {filename_base}.pdf")
        
        # This check now relies on whether legend_types was successfully populated
        if not self.legend_types: # If set is empty, no valid legend data was loaded
            print("Warning: No valid legend data loaded. Instrument tags will NOT be filtered by type.")


        try:
            # Step 1: Convert PDF to image
            with open(pid_file_path, 'rb') as f:
                pdf_content = f.read()

            if self.poppler_path:
                images = convert_from_bytes(pdf_content, dpi=PDF_DPI, poppler_path=self.poppler_path)
            else:
                images = convert_from_bytes(pdf_content, dpi=PDF_DPI)
            
            if not images:
                print(f"Error: No images converted from {filename_base}.pdf")
                return pd.DataFrame()

            pil_image = images[0]
            if self.debug_mode:
                debug_img_path = os.path.join(self.debug_output_folder, f"{filename_base}_page1.jpg")
                pil_image.save(debug_img_path, "JPEG")
                print(f"Saved debug image: {debug_img_path}")

            cv_image = np.array(pil_image)
            cv_image_rgb = cv_image[:, :, ::-1].copy()

            # Step 2: Detect circles using OpenCV
            gray_image = cv2.cvtColor(cv_image_rgb, cv2.COLOR_BGR2GRAY)
            blurred_image = cv2.medianBlur(gray_image, 5) 
            
            circles = cv2.HoughCircles(
                blurred_image,
                cv2.HOUGH_GRADIENT,
                dp=HOUGH_DP,           
                minDist=HOUGH_MIN_DIST,
                param1=HOUGH_PARAM1,   
                param2=HOUGH_PARAM2,   
                minRadius=HOUGH_MIN_RADIUS,
                maxRadius=HOUGH_MAX_RADIUS 
            )
            
            detected_instruments = []
            if circles is not None:
                circles = np.uint16(np.around(circles))
                print(f"Detected {len(circles[0])} circles in {filename_base}.pdf.")

                pil_image_draw = pil_image.copy()
                draw = ImageDraw.Draw(pil_image_draw)
                
                try:
                    font = ImageFont.truetype("arial.ttf", 40)
                except IOError:
                    font = ImageFont.load_default()

                raw_ocr_data = [] 

                for i, c in enumerate(circles[0]):
                    center_x, center_y, radius = c[0], c[1], c[2]
                    
                    margin = int(radius * OCR_ROI_MARGIN_FACTOR)
                    x1 = max(0, center_x - margin)
                    y1 = max(0, center_y - margin)
                    x2 = min(pil_image.width, center_x + margin)
                    y2 = min(pil_image.height, center_y + margin)
                    
                    roi_image = pil_image.crop((x1, y1, x2, y2))

                    if self.debug_mode:
                        roi_debug_path = os.path.join(self.debug_output_folder, f"{filename_base}_ROI_{i+1}.jpg")
                        roi_image.save(roi_debug_path, "JPEG")

                    img_byte_arr = io.BytesIO()
                    roi_image.save(img_byte_arr, format='JPEG')
                    img_byte_arr = img_byte_arr.getvalue()
                    
                    image = vision.Image(content=img_byte_arr)
                    response = self.vision_client.text_detection(image=image)
                    texts = response.text_annotations

                    if texts:
                        for text_annotation in texts[1:]: 
                            raw_ocr_data.append({
                                'Circle_ID': i,
                                'Circle_X': center_x,
                                'Circle_Y': center_y,
                                'Circle_Radius': radius,
                                'Text': text_annotation.description,
                                'BoundingBox': [(v.x, v.y) for v in text_annotation.bounding_poly.vertices]
                            })
            else:
                print(f"No circles detected in {filename_base}.pdf with current parameters.")
                return pd.DataFrame() 

            if self.debug_mode:
                raw_ocr_df = pd.DataFrame(raw_ocr_data)
                raw_ocr_csv_path = os.path.join(self.debug_output_folder, f"{filename_base}_raw_ocr_output.csv")
                raw_ocr_df.to_csv(raw_ocr_csv_path, index=False)
                print(f"Saved raw OCR output to: {raw_ocr_csv_path}")

            # --- Post-OCR Processing: Grouping, Concatenating, and Validating ---
            final_instruments_data = []

            grouped_ocr_results = {}
            for row in raw_ocr_data:
                circle_id = row['Circle_ID']
                if circle_id not in grouped_ocr_results:
                    grouped_ocr_results[circle_id] = {
                        'Circle_X': row['Circle_X'],
                        'Circle_Y': row['Circle_Y'],
                        'Circle_Radius': row['Circle_Radius'],
                        'Texts': []
                    }
                
                bbox_vertices = row['BoundingBox']
                if bbox_vertices:
                    # Calculate average Y for sorting
                    avg_y = sum(v[1] for v in bbox_vertices) / len(bbox_vertices)
                else:
                    avg_y = row['Circle_Y'] # Fallback if bbox is empty

                grouped_ocr_results[circle_id]['Texts'].append({
                    'Text': row['Text'],
                    'Avg_Y': avg_y
                })
            
            for circle_id, circle_data in grouped_ocr_results.items():
                sorted_texts = sorted(circle_data['Texts'], key=lambda x: x['Avg_Y'])
                concatenated_text = TEXT_CONCAT_SEPARATOR.join([t['Text'] for t in sorted_texts]).strip()

                if not concatenated_text:
                    continue

                instrument_type = concatenated_text.split(TEXT_CONCAT_SEPARATOR)[0].strip().upper()

                # Validate against legend ONLY if legend data is loaded
                is_valid_instrument = False
                if self.legend_types and instrument_type in self.legend_types:
                    is_valid_instrument = True
                elif not self.legend_types: # No legend uploaded, so don't filter
                    is_valid_instrument = True # Treat as valid if no legend provided

                if is_valid_instrument:
                    final_instruments_data.append({
                        'P&ID_Filename': filename_base + ".pdf",
                        'Instrument_Tag': concatenated_text,
                        'Instrument_Type': instrument_type,
                        'X_Coordinate': circle_data['Circle_X'],
                        'Y_Coordinate': circle_data['Circle_Y'],
                        'Radius': circle_data['Circle_Radius']
                    })
                    
                    x, y, r = circle_data['Circle_X'], circle_data['Circle_Y'], circle_data['Circle_Radius']
                    draw.ellipse((x - r, y - r, x + r, y + r), outline=(0, 255, 0), width=5)
                    
                    # Corrected text drawing logic for valid instruments
                    text_bbox_at_origin = font.getbbox(concatenated_text)
                    text_width = text_bbox_at_origin[2] - text_bbox_at_origin[0]
                    text_height = text_bbox_at_origin[3] - text_bbox_at_origin[1]

                    text_draw_x = x - text_width / 2
                    text_draw_y = y + r + 10

                    text_draw_x = max(0, text_draw_x)
                    text_draw_y = max(0, text_draw_y)

                    draw.text((text_draw_x, text_draw_y), concatenated_text, fill=(0, 255, 0), font=font)

                elif self.debug_mode:
                    x, y, r = circle_data['Circle_X'], circle_data['Circle_Y'], circle_data['Circle_Radius']
                    draw.ellipse((x - r, y - r, x + r, y + r), outline=(255, 0, 0), width=5)
                    
                    text_to_draw = f"NO MATCH: {concatenated_text}" if self.legend_types else f"NO LEGEND: {concatenated_text}"
                    
                    # Corrected text drawing logic for debug mode (no match)
                    text_bbox_at_origin = font.getbbox(text_to_draw)
                    text_width = text_bbox_at_origin[2] - text_bbox_at_origin[0]
                    text_height = text_bbox_at_origin[3] - text_bbox_at_origin[1]

                    text_draw_x = x - text_width / 2
                    text_draw_y = y + r + 10

                    text_draw_x = max(0, text_draw_x)
                    text_draw_y = max(0, text_draw_y)

                    draw.text((text_draw_x, text_draw_y), text_to_draw, fill=(255, 0, 0), font=font)


            final_instruments_df = pd.DataFrame(final_instruments_data)

            if self.debug_mode and circles is not None:
                debug_circles_img_path = os.path.join(self.debug_output_folder, f"{filename_base}_circles_detected.jpg")
                pil_image_draw.save(debug_circles_img_path, "JPEG")
                print(f"Saved debug image with detected circles: {debug_circles_img_path}")

            return final_instruments_df

        except Exception as e:
            print(f"Error processing {filename_base}.pdf: {e}")
            import traceback
            traceback.print_exc() # Print full traceback for debugging
            return pd.DataFrame()