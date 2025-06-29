import os

# --- Google Cloud Vision API Configuration ---
# IMPORTANT: Replace with the actual FULL PATH to your Google Service Account JSON key file
# Examples:
# Windows: r"C:\path\to\your\service-account-key.json"
# Linux/macOS: "/home/youruser/path/to/your/service-account-key.json"
# The 'r' before the string on Windows means "raw string" and helps with backslashes.
#GOOGLE_APPLICATION_CREDENTIALS_PATH = r"C:\Instrument Extractor\pro-sylph-460317-k6-72abff84fb8a.json"
GOOGLE_APPLICATION_CREDENTIALS_PATH = "/Users/prashanththipparthi/Auto_Draw/autodraw-464410-92725d562783.json"
# --- Poppler Path Configuration ---
# IMPORTANT: Replace with the actual FULL PATH to the 'bin' directory of your Poppler installation
# On Windows, this is usually something like: r"C:\path\to\poppler\po3.10\Library\bin"
# On macOS/Linux, if you installed via brew/apt, it's often not needed as it's in your PATH.
# If you get 'pdfinfo' or 'pdftocairo' not found errors, specify it here.
POPPLER_PATH = r"/opt/homebrew/opt/poppler/bin" # <-- UPDATE THIS LINE!

# --- Flask App Configuration ---
UPLOAD_FOLDER = 'uploads' # Temporary folder for uploads (Flask handles cleanup for request.files)
ALLOWED_EXTENSIONS = {'pdf', 'xlsx', 'csv'} # File types allowed for upload
# IMPORTANT: Change this to a strong, random key in production! It's for security.
SECRET_KEY = 'AIzaSyBh4I3rfb1AHjHkgVu4ZBUC4g2LbymQ194'

# --- OpenCV HoughCircles Parameters ---
# These parameters are crucial for finding circles on your P&IDs.
# You might need to change these numbers later if the app doesn't find circles well.
# Briefly:
# dp: Controls how accurately the circle centers are located. 1 is good.
# minDist: Minimum distance allowed between the centers of two detected circles. Prevents many circles for one.
# param1: Higher threshold for the Canny edge detector (used internally).
# param2: Accumulator threshold. Smaller values find more circles (even weak ones), larger values find fewer but stronger ones.
# minRadius, maxRadius: The smallest and largest circle sizes to look for.
HOUGH_DP = 1        
HOUGH_MIN_DIST = 50 
HOUGH_PARAM1 = 100  
HOUGH_PARAM2 = 30   
HOUGH_MIN_RADIUS = 20 
HOUGH_MAX_RADIUS = 60 

# --- Image Processing Parameters ---
PDF_DPI = 300 # DPI (Dots Per Inch) for converting PDF to image. Higher DPI = better OCR, but slower.
# Multiplier for circle radius to define the box size for OCR.
# A value of 1.5 means the box will be 1.5 times the radius on each side of the center.
OCR_ROI_MARGIN_FACTOR = 1.8 

# --- Text processing parameters ---
TEXT_CONCAT_SEPARATOR = '-' # Separator for joining text parts within a circle (e.g., P-573P-01)

# --- Debugging Configuration ---
# Set to True to enable saving intermediate images and OCR outputs for debugging
DEBUG_MODE = True
# Folder to save debug outputs. This will be created inside your Auto_Draw project.
DEBUG_OUTPUT_FOLDER = 'debug_outputs'