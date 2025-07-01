import os
import pandas as pd
from flask import Flask, request, render_template, redirect, url_for, send_from_directory, send_file
from werkzeug.utils import secure_filename
from instrument_processor import InstrumentProcessor

# Import configuration settings
from config import (
    UPLOAD_FOLDER, ALLOWED_EXTENSIONS, GOOGLE_APPLICATION_CREDENTIALS_PATH, POPPLER_PATH,
    PDF_DPI, HOUGH_DP, HOUGH_MIN_DIST, HOUGH_PARAM1, HOUGH_PARAM2,
    HOUGH_MIN_RADIUS, HOUGH_MAX_RADIUS, # Re-added for initial broad detection
    OCR_ROI_MARGIN_FACTOR, TEXT_CONCAT_SEPARATOR,
    DEBUG_MODE, DEBUG_OUTPUT_FOLDER,
    ANCHOR_MIN_COUNT, RADIUS_TOLERANCE_PERCENT # New dynamic radius calibration parameters
)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload size

# Ensure upload and debug folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DEBUG_OUTPUT_FOLDER, exist_ok=True)

# Initialize the InstrumentProcessor (only once per Flask app instance)
processor = InstrumentProcessor()

def allowed_file(filename):
    """Checks if a file's extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Renders the main upload page."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handles file uploads and initiates the processing."""
    # Check if files were uploaded at all
    if 'pid_file' not in request.files:
        return redirect(request.url)
    
    # Get lists of P&ID files and the single legend file
    pid_files = request.files.getlist('pid_file')
    legend_file = request.files.get('legend_file')

    # Radius inputs from the form are no longer needed here as they are handled
    # dynamically within instrument_processor.py based on the new two-level strategy.

    # Basic validation: ensure at least one P&ID file is selected
    if not pid_files or all(f.filename == '' for f in pid_files):
        return render_template('index.html', message='No P&ID files selected for upload.', error=True)

    # Limit the number of P&ID files for performance
    if len(pid_files) > 50:
        return render_template('index.html', message='Maximum 50 P&ID files allowed per upload.', error=True)

    # --- Process legend file first (if provided) ---
    legend_df = pd.DataFrame()
    if legend_file and legend_file.filename != '':
        legend_filename = secure_filename(legend_file.filename)
        legend_file_path = os.path.join(app.config['UPLOAD_FOLDER'], legend_filename)
        legend_file.save(legend_file_path)
        print(f"Legend file saved to: {legend_file_path}")

        lower_legend_filename = legend_filename.lower()
        if lower_legend_filename.endswith(('.xlsx', '.xls')):
            try:
                # Read Excel file, assuming first sheet for now
                legend_df = pd.read_excel(legend_file_path)
            except Exception as e:
                print(f"ERROR: Could not read Excel file {legend_filename}: {e}")
        elif lower_legend_filename.endswith('.csv'):
            try:
                # Read CSV file
                legend_df = pd.read_csv(legend_file_path)
            except Exception as e:
                print(f"ERROR: Could not read CSV file {legend_filename}: {e}")
        else:
            print("Unsupported legend file type. Please upload an Excel (.xlsx, .xls) or CSV (.csv) file.")
        
        # Ensure column names are consistent: strip whitespace and rename 'Instrument Type' to 'Instrument_Type'
        if not legend_df.empty:
            legend_df.columns = [col.strip() for col in legend_df.columns] 
            if 'Instrument Type' in legend_df.columns:
                legend_df = legend_df.rename(columns={'Instrument Type': 'Instrument_Type'})
        
        # Debugging output for legend DataFrame structure
        print(f"DEBUG: legend_df is empty: {legend_df.empty}")
        print(f"DEBUG: legend_df shape: {legend_df.shape}")
        print(f"DEBUG: legend_df columns: {list(legend_df.columns) if not legend_df.empty else 'No columns'}")
        
        # Load the legend data into the processor
        processor.load_legend_data(legend_df) 
    else:
        print("No legend file uploaded. Instrument tags will not be filtered by type, and descriptions cannot be mapped.")
        processor.load_legend_data(pd.DataFrame()) # Pass an empty DataFrame if no legend

    # --- Process multiple P&ID files and collect data ---
    all_instruments_data = [] # List to hold DataFrames from each P&ID
    processed_count = 0
    failed_files = [] # To keep track of files that couldn't be processed

    for pid_file in pid_files:
        if pid_file and allowed_file(pid_file.filename):
            try:
                pid_filename = secure_filename(pid_file.filename)
                pid_file_path = os.path.join(app.config['UPLOAD_FOLDER'], pid_filename)
                pid_file.save(pid_file_path) # Save P&ID file temporarily
                print(f"Starting processing for P&ID: {pid_file_path}")

                filename_base = os.path.splitext(pid_filename)[0]
                
                # Call the instrument processor without radius parameters; it handles dynamically
                final_instruments_df = processor.process_pid_file(pid_file_path, filename_base)

                if not final_instruments_df.empty:
                    final_instruments_df['Source_PID_File'] = pid_filename # Add original filename to output
                    
                    # Sort instruments by Y then X coordinate for a logical reading order
                    if 'Y_Coordinate' in final_instruments_df.columns and 'X_Coordinate' in final_instruments_df.columns:
                        final_instruments_df = final_instruments_df.sort_values(
                            by=['Y_Coordinate', 'X_Coordinate'], ascending=[True, True]
                        ).reset_index(drop=True)
                        print(f"Instruments sorted by position for {pid_filename}.")
                    else:
                        print(f"WARNING: Positional columns ('Y_Coordinate', 'X_Coordinate') not found in DataFrame for {pid_filename}. Skipping positional sort. Please ensure instrument_processor.py returns these columns.")

                    all_instruments_data.append(final_instruments_df)
                    processed_count += 1
                    print(f"Instruments extracted from {pid_filename}.")
                else:
                    print(f"Processing completed for {pid_filename}, but no instruments were detected.")
                    failed_files.append(pid_filename + " (no instruments detected)")

            except Exception as e:
                print(f"ERROR: Could not process P&ID file {pid_file.filename}: {e}")
                failed_files.append(pid_file.filename + f" (Error: {e})")
            finally:
                # Clean up: remove the temporary P&ID file
                if os.path.exists(pid_file_path):
                    os.remove(pid_file_path)

    # --- Combine all processed data into a single CSV ---
    if all_instruments_data:
        combined_df = pd.concat(all_instruments_data, ignore_index=True)

        # Merge descriptions from legend if legend was provided and valid
        if not legend_df.empty and 'Instrument_Type' in legend_df.columns and 'Description' in legend_df.columns:
            legend_for_merge = legend_df[['Instrument_Type', 'Description']].drop_duplicates()
            
            # Use left merge to keep all detected instruments and add description if a match is found
            combined_df = pd.merge(
                combined_df, 
                legend_for_merge, 
                on='Instrument_Type', 
                how='left'
            )
            print("Description column mapped from legend to combined output.")
            
            # Reorder columns to place 'Description' next to 'Instrument_Type'
            cols = combined_df.columns.tolist()
            if 'Description' in cols and 'Instrument_Type' in cols:
                desc_idx = cols.index('Description')
                type_idx = cols.index('Instrument_Type')
                if desc_idx != type_idx + 1: # Only move if not already next to it
                    cols.insert(type_idx + 1, cols.pop(desc_idx))
                    combined_df = combined_df[cols]
        else:
            print("WARNING: Legend not suitable for description mapping (missing 'Instrument_Type' or 'Description' columns, or legend is empty). Skipping description merge.")

        # Define output CSV filename and path
        output_csv_filename = "combined_pid_instruments.csv" 
        output_csv_path = os.path.join(app.config['UPLOAD_FOLDER'], output_csv_filename)
        combined_df.to_csv(output_csv_path, index=False) # Save the combined data
        print(f"All instruments combined and saved to: {output_csv_path}")

        # Prepare success message for the user
        message = f"Successfully processed {processed_count} P&ID file(s) and combined their data into one CSV. "
        if failed_files:
            message += f"Note: Some files had issues or no instruments: {', '.join(failed_files)}. "
        message += "Your combined results are ready for download."

        return render_template(
            'index.html', 
            message=message, 
            output_file=output_csv_filename # Pass filename to the template for download link
        )
    else:
        # Handle case where no files were successfully processed
        error_message = "No P&ID files were successfully processed or yielded instruments."
        if failed_files:
            error_message += f" Details: {', '.join(failed_files)}."
        return render_template('index.html', message=error_message, error=True)

    # This return is a fallback and should ideally not be reached
    return render_template('index.html', message='Invalid file type or no file selected.', error=True)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serves the uploaded/generated files for download."""
    return send_from_directory(
        app.config['UPLOAD_FOLDER'],
        filename,
        as_attachment=True,         # Force download
        download_name=filename      # Suggest original filename for download
    )

if __name__ == '__main__':
    # Run the Flask development server
    app.run(debug=True)