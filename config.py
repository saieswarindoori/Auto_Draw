import os

# --- Google Cloud Vision API Configuration ---
# IMPORTANT: Replace with the actual path to your Google Cloud service account key JSON file
GOOGLE_APPLICATION_CREDENTIALS_PATH = r"C:\Users\idm280865\OneDrive - Bilfinger\Desktop\Working\Auto_Draw\autodraw-464410-3cb4dae6350d.json"


# --- Poppler Path Configuration ---
# IMPORTANT: Update this line with the correct path to your Poppler bin directory
# On macOS with Homebrew, it's typically /opt/homebrew/opt/poppler/bin
# On Windows, it would be the path to the 'bin' folder within your Poppler installation (e.g., r"C:\path\to\poppler\bin")


POPPLER_PATH = r"C:\Users\idm280865\poppler-24.02.0\poppler-24.02.0\Library\bin"


# --- Flask App Configuration ---
UPLOAD_FOLDER = 'uploads' 
ALLOWED_EXTENSIONS = {'pdf', 'xlsx', 'csv'} 
SECRET_KEY = 'AIzaSyCAbSXGwvYBJD2lHVp8bYhFo9DmtRD22R4' # Replace with a strong, unique secret key

# --- OpenCV HoughCircles Parameters (Broad Defaults for Initial Scan) ---
# These are used for the broad initial search in Level 1 to find anchor circles
HOUGH_DP = 1        
HOUGH_MIN_DIST = 50 
HOUGH_PARAM1 = 100  
HOUGH_PARAM2 = 30  
HOUGH_MIN_RADIUS = 10  # Broad minimum, to capture potentially very small circles
HOUGH_MAX_RADIUS = 100 # Broad maximum, to capture potentially very large circles

# config.py

# ... (previous configurations) ...

# OCR Validation Parameters for Instrument Tags
OCR_MIN_CHARS_PER_ROW = 2  # Minimum characters expected per line of an instrument tag
OCR_MAX_CHARS_PER_ROW = 5  # Maximum characters expected per line of an instrument tag
OCR_MAX_TAG_ROWS = 4       # Maximum number of lines (rows) expected in an instrument tag
OCR_Y_TOLERANCE = 5        # Y-axis tolerance for grouping OCR text into lines (in pixels)

# ... (rest of your config.py) ...

# --- Image Processing Parameters ---
PDF_DPI = 300 # Dots per inch for PDF conversion (higher DPI means better OCR, but slower processing)
OCR_ROI_MARGIN_FACTOR = 1 # Multiplier for radius to define the OCR ROI (e.g., 1.5x the radius)

# --- Text processing parameters ---
TEXT_CONCAT_SEPARATOR = '-' # Separator for combining lines of text in an instrument tag

# --- Debugging Configuration ---
DEBUG_MODE = True # Set to False to disable debug output
DEBUG_OUTPUT_FOLDER = 'debug_outputs'

# --- Dynamic Radius Calibration Parameters for Level 1 ---
ANCHOR_MIN_COUNT = 1 # The minimum number of suitable circles needed to determine a dynamic radius
RADIUS_TOLERANCE_PERCENT = 0.15 # +/- 15% from the average radius for the dynamic range (e.g., 0.15 for +/-15%)

# --- Grid-based Search Parameters for Level 1 Calibration ---
PID_GRID_ROWS = 4 # Number of rows to divide the P&ID image into
PID_GRID_COLS = 5 # Number of columns to divide the P&ID image into
# This defines the order in which grid parts (1-indexed) are searched for anchor circles.
# The idea is to prioritize central areas where instruments are more commonly placed.
PID_SEARCH_ORDER = [
    7, 8, 9, 13, 12, # Core central zones (e.g., parts in rows 2 & 3, columns 2-4)
    6, 10, 11, 14, 15, # Adjacent central zones
    2, 3, 4, 1, 5,    # Top row
    17, 18, 19, 16, 20 # Bottom row
]