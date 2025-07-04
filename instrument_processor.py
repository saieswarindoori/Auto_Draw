# instrument_processor.py
import os
import pandas as pd
import numpy as np
from pdf2image import convert_from_bytes
from google.cloud import vision
from PIL import Image, ImageDraw, ImageFont
import io
import math
import cv2 # <--- ADDED THIS IMPORT

# Import configuration settings
from config import (
    GOOGLE_APPLICATION_CREDENTIALS_PATH, POPPLER_PATH, PDF_DPI,
    HOUGH_DP, HOUGH_MIN_DIST, HOUGH_PARAM1, HOUGH_PARAM2,
    HOUGH_MIN_RADIUS, HOUGH_MAX_RADIUS,
    OCR_ROI_MARGIN_FACTOR, TEXT_CONCAT_SEPARATOR,
    DEBUG_MODE, DEBUG_OUTPUT_FOLDER,
    ANCHOR_MIN_COUNT, RADIUS_TOLERANCE_PERCENT,
    PID_GRID_ROWS, PID_GRID_COLS, PID_SEARCH_ORDER,
    OCR_MIN_CHARS_PER_ROW, OCR_MAX_CHARS_PER_ROW, OCR_MAX_TAG_ROWS, OCR_Y_TOLERANCE
)

# Import the new modules for Level 1 and Level 2 processing
from level1_calibration import calibrate_radius_range
from level2_extraction import extract_instruments

class InstrumentProcessor:
    def __init__(self):
        # Set Google Cloud credentials environment variable
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = GOOGLE_APPLICATION_CREDENTIALS_PATH
        # Initialize Google Cloud Vision client
        self.vision_client = vision.ImageAnnotatorClient()
        print("Google Vision Client initialized successfully.")

        # Set Poppler path for PDF conversion
        self.poppler_path = POPPLER_PATH
        print(f"InstrumentProcessor initialized with Poppler Path: {self.poppler_path if self.poppler_path else 'Not explicitly set (relying on PATH)'}")

        # Initialize legend data structures
        self.legend_df = pd.DataFrame()
        self.legend_types = set()
        print(f"Legend data initialized as empty. Please upload a legend file.")

        # Set debug mode and create debug output folder if enabled
        self.debug_mode = DEBUG_MODE
        self.debug_output_folder = DEBUG_OUTPUT_FOLDER
        if self.debug_mode and not os.path.exists(self.debug_output_folder):
            os.makedirs(self.debug_output_folder)
        print(f"Debug mode is {'ON' if self.debug_mode else 'OFF'.upper()}. Outputs will be saved to: {self.debug_output_folder}")

        # Bundle all relevant config parameters to pass to sub-modules
        # This ensures that sub-modules get all necessary configuration without direct config.py imports
        self.config_params = {
            'PDF_DPI': PDF_DPI,
            'HOUGH_DP': HOUGH_DP,
            'HOUGH_MIN_DIST': HOUGH_MIN_DIST,
            'HOUGH_PARAM1': HOUGH_PARAM1,
            'HOUGH_PARAM2': HOUGH_PARAM2,
            'HOUGH_MIN_RADIUS': HOUGH_MIN_RADIUS,
            'HOUGH_MAX_RADIUS': HOUGH_MAX_RADIUS,
            'OCR_ROI_MARGIN_FACTOR': OCR_ROI_MARGIN_FACTOR,
            'TEXT_CONCAT_SEPARATOR': TEXT_CONCAT_SEPARATOR,
            'ANCHOR_MIN_COUNT': ANCHOR_MIN_COUNT,
            'RADIUS_TOLERANCE_PERCENT': RADIUS_TOLERANCE_PERCENT,
            'PID_GRID_ROWS': PID_GRID_ROWS,
            'PID_GRID_COLS': PID_GRID_COLS,
            'PID_SEARCH_ORDER': PID_SEARCH_ORDER,
            'OCR_MIN_CHARS_PER_ROW': OCR_MIN_CHARS_PER_ROW,
            'OCR_MAX_CHARS_PER_ROW': OCR_MAX_CHARS_PER_ROW,
            'OCR_MAX_TAG_ROWS': OCR_MAX_TAG_ROWS,
            'OCR_Y_TOLERANCE': OCR_Y_TOLERANCE
        }

        # Bundle debug parameters to pass to sub-modules
        self.debug_params = {
            'DEBUG_MODE': DEBUG_MODE,
            'DEBUG_OUTPUT_FOLDER': DEBUG_OUTPUT_FOLDER
        }


    def load_legend_data(self, legend_df):
        """
        Loads and processes the instrument legend data.
        It expects a DataFrame with an 'Instrument_Type' column and optionally a 'Description' column.
        """
        self.legend_df = legend_df

        # Check for the standardized column name 'Instrument_Type'
        if 'Instrument_Type' in self.legend_df.columns:
            # Convert instrument types to uppercase and store as a set for quick lookup
            self.legend_types = set(self.legend_df['Instrument_Type'].astype(str).str.upper())
            # Ensure 'Description' column exists, add an empty one if not to prevent KeyError later
            if 'Description' not in self.legend_df.columns:
                self.legend_df['Description'] = ''
                print("Warning: 'Description' column not found in legend, adding an empty one.")
            print(f"Legend data loaded using 'Instrument_Type' and 'Description' columns.")
            print(f"Loaded {len(self.legend_types)} legend types: {sorted(list(self.legend_types))}")
        else:
            # If the required column is missing, clear legend data and warn
            self.legend_types = set()
            self.legend_df = pd.DataFrame() # Ensure it's an empty DataFrame if required columns are missing
            print("Warning: 'Instrument_Type' column not found in legend. No types loaded for validation.")

    def process_pid_file(self, pid_file_path, filename_base):
        """
        Main method to process a single P&ID file.
        It orchestrates the Level 1 calibration and Level 2 instrument extraction.
        """
        print(f"Starting processing for P&ID: {filename_base}.pdf")

        # Determine if legend is available for calibration and type filtering
        if not self.legend_types:
            print("Warning: No valid legend data loaded. Cannot use legend for dynamic radius calibration or type filtering. Falling back to default broad radius for all circles.")
            legend_available_for_calibration = False
        else:
            legend_available_for_calibration = True

        try:
            # Convert PDF to image using pdf2image library
            with open(pid_file_path, 'rb') as f:
                pdf_content = f.read()

            # Use poppler_path if configured, otherwise rely on system PATH
            if self.poppler_path:
                images = convert_from_bytes(pdf_content, dpi=self.config_params['PDF_DPI'], poppler_path=self.poppler_path)
            else:
                images = convert_from_bytes(pdf_content, dpi=self.config_params['PDF_DPI'])

            # Handle case where no images are converted
            if not images:
                print(f"Error: No images converted from {filename_base}.pdf")
                return pd.DataFrame()

            pil_image = images[0] # Process only the first page for now
            # Save debug image of the first page if debug mode is on
            if self.debug_mode:
                debug_img_path = os.path.join(self.debug_output_folder, f"{filename_base}_page1.jpg")
                pil_image.save(debug_img_path, "JPEG")
                print(f"Saved debug image: {debug_img_path}")

            # Convert PIL image to OpenCV format for image processing
            cv_image = np.array(pil_image)
            cv_image_rgb = cv_image[:, :, ::-1].copy() # Convert RGB to BGR for OpenCV
            print(f"DEBUG: cv_image_rgb shape: {cv_image_rgb.shape}")

            # Pre-process image for circle detection: convert to grayscale and apply blur
            gray_image = cv2.cvtColor(cv_image_rgb, cv2.COLOR_BGR2GRAY)
            print(f"DEBUG: gray_image shape: {gray_image.shape}")

            blurred_image = cv2.medianBlur(gray_image, 5) # Apply median blur to reduce noise
            print(f"DEBUG: blurred_image shape: {blurred_image.shape}")

            # Validate blurred image dimensions
            if blurred_image.shape[0] == 0 or blurred_image.shape[1] == 0:
                print("ERROR: Blurred image has zero dimensions. Cannot proceed with circle detection.")
                return pd.DataFrame()

            # Prepare PIL image for drawing debug annotations if debug mode is on
            if self.debug_mode:
                pil_image_draw = pil_image.copy()
                try:
                    font = ImageFont.truetype("arial.ttf", 30) # Attempt to load Arial font
                except IOError:
                    font = ImageFont.load_default() # Fallback to default font if Arial not found
            else:
                pil_image_draw = None # No drawing needed if debug mode is off
                font = None

            # --- LEVEL 1: Identify the Correct Circle (Calibration Phase) ---
            print(f"Calling Level 1 Calibration for {filename_base}.pdf")
            # Call the calibration function from level1_calibration.py
            dynamic_min_radius, dynamic_max_radius = calibrate_radius_range(
                pil_image=pil_image_draw, # Pass the image for drawing if in debug mode
                blurred_image=blurred_image,
                vision_client=self.vision_client,
                legend_types=self.legend_types,
                config_params=self.config_params,
                debug_params=self.debug_params,
                font=font # Pass font for drawing
            )
            print(f"Level 1 Calibration returned dynamic radius range: {dynamic_min_radius}-{dynamic_max_radius}")

            # --- LEVEL 2: List the Instruments (Full Detection Phase) ---
            print(f"Calling Level 2 Extraction for {filename_base}.pdf")
            # Call the instrument extraction function from level2_extraction.py
            final_instruments_df = extract_instruments(
                pil_image=pil_image_draw, # Pass the image for drawing if in debug mode
                blurred_image=blurred_image,
                vision_client=self.vision_client,
                dynamic_min_radius=dynamic_min_radius,
                dynamic_max_radius=dynamic_max_radius,
                legend_df=self.legend_df,
                legend_types=self.legend_types,
                filename_base=filename_base,
                config_params=self.config_params,
                debug_params=self.debug_params,
                font=font # Pass font for drawing
            )
            print(f"Level 2 Extraction completed for {filename_base}.pdf")

            # --- Deduplicate based on rounded coordinates and tag ---
            if not final_instruments_df.empty:
                # Round coordinates to integers for more robust deduplication
                final_instruments_df['Rounded_X'] = final_instruments_df['X_Coordinate'].round(0).astype(int)
                final_instruments_df['Rounded_Y'] = final_instruments_df['Y_Coordinate'].round(0).astype(int)

                # Deduplicate, keeping the first occurrence.
                # The primary unique identifier for an instrument should be its tag and location.
                initial_rows = len(final_instruments_df)
                final_instruments_df.drop_duplicates(
                    subset=['Instrument_Tag', 'Rounded_X', 'Rounded_Y'],
                    keep='first',
                    inplace=True
                )
                rows_after_dedup = len(final_instruments_df)
                if initial_rows > rows_after_dedup:
                    print(f"Deduplication removed {initial_rows - rows_after_dedup} duplicate instrument entries.")

                # Drop the temporary rounded coordinate columns
                final_instruments_df.drop(columns=['Rounded_X', 'Rounded_Y'], inplace=True)


            # Save the final debug image with all detected circles and annotations
            if self.debug_mode:
                debug_circles_img_path = os.path.join(self.debug_output_folder, f"{filename_base}_circles_detected.jpg")
                pil_image_draw.save(debug_circles_img_path, "JPEG")
                print(f"Saved debug image with detected circles: {debug_circles_img_path}")

            return final_instruments_df

        except Exception as e:
            # Catch and log any exceptions during processing
            print(f"Error processing {filename_base}.pdf: {e}")
            import traceback
            traceback.print_exc() # Print full traceback for detailed debugging
            return pd.DataFrame() # Return empty DataFrame on error

