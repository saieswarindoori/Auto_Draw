import os
import pandas as pd
from flask import Flask, request, render_template, redirect, url_for, send_from_directory, send_file
from werkzeug.utils import secure_filename
from instrument_processor import InstrumentProcessor
# Removed zipfile and shutil imports as they are no longer needed for single CSV output

# Import configuration settings
from config import (
    UPLOAD_FOLDER, ALLOWED_EXTENSIONS, GOOGLE_APPLICATION_CREDENTIALS_PATH, POPPLER_PATH,
    PDF_DPI, HOUGH_DP, HOUGH_MIN_DIST, HOUGH_PARAM1, HOUGH_PARAM2,
    HOUGH_MIN_RADIUS, HOUGH_MAX_RADIUS, OCR_ROI_MARGIN_FACTOR, TEXT_CONCAT_SEPARATOR,
    DEBUG_MODE, DEBUG_OUTPUT_FOLDER
)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload size

# Ensure upload and debug folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DEBUG_OUTPUT_FOLDER, exist_ok=True)

processor = InstrumentProcessor()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'pid_file' not in request.files:
        return redirect(request.url)
    
    pid_files = request.files.getlist('pid_file')
    legend_file = request.files.get('legend_file')

    if not pid_files or all(f.filename == '' for f in pid_files):
        return render_template('index.html', message='No P&ID files selected for upload.', error=True)

    if len(pid_files) > 50:
        return render_template('index.html', message='Maximum 50 P&ID files allowed per upload.', error=True)

    # --- Process legend file first ---
    legend_df = pd.DataFrame()
    if legend_file and legend_file.filename != '':
        legend_filename = secure_filename(legend_file.filename)
        legend_file_path = os.path.join(app.config['UPLOAD_FOLDER'], legend_filename)
        legend_file.save(legend_file_path)
        print(f"Legend file saved to: {legend_file_path}")

        lower_legend_filename = legend_filename.lower()
        if lower_legend_filename.endswith(('.xlsx', '.xls')):
            try:
                legend_df = pd.read_excel(legend_file_path)
            except Exception as e:
                print(f"ERROR: Could not read Excel file {legend_filename}: {e}")
        elif lower_legend_filename.endswith('.csv'):
            try:
                legend_df = pd.read_csv(legend_file_path)
            except Exception as e:
                print(f"ERROR: Could not read CSV file {legend_filename}: {e}")
        else:
            print("Unsupported legend file type. Please upload an Excel (.xlsx, .xls) or CSV (.csv) file.")
        
        print(f"DEBUG: legend_df is empty: {legend_df.empty}")
        print(f"DEBUG: legend_df shape: {legend_df.shape}")
        print(f"DEBUG: legend_df columns: {list(legend_df.columns) if not legend_df.empty else 'No columns'}")
        
        processor.load_legend_data(legend_df)
    else:
        print("No legend file uploaded. Instrument tags will not be filtered by type.")
        processor.load_legend_data(pd.DataFrame())

    # --- Process multiple P&ID files and collect data ---
    all_instruments_data = [] # List to hold DataFrames from each P&ID
    processed_count = 0
    failed_files = []

    for pid_file in pid_files:
        if pid_file and allowed_file(pid_file.filename):
            try:
                pid_filename = secure_filename(pid_file.filename)
                pid_file_path = os.path.join(app.config['UPLOAD_FOLDER'], pid_filename)
                pid_file.save(pid_file_path)
                print(f"Starting processing for P&ID: {pid_file_path}")

                filename_base = os.path.splitext(pid_filename)[0]
                final_instruments_df = processor.process_pid_file(pid_file_path, filename_base)

                if not final_instruments_df.empty:
                    # Add a column to identify the source PDF for each instrument
                    final_instruments_df['Source_PID_File'] = pid_filename
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
                # Clean up the original uploaded PDF file from UPLOAD_FOLDER after processing
                if os.path.exists(pid_file_path):
                    os.remove(pid_file_path)

    # --- Combine all data into a single CSV ---
    if all_instruments_data:
        combined_df = pd.concat(all_instruments_data, ignore_index=True)
        output_csv_filename = "combined_pid_instruments.csv" # Name for the single output CSV
        output_csv_path = os.path.join(app.config['UPLOAD_FOLDER'], output_csv_filename)
        combined_df.to_csv(output_csv_path, index=False)
        print(f"All instruments combined and saved to: {output_csv_path}")

        # Construct the success message
        message = f"Successfully processed {processed_count} P&ID file(s) and combined their data into one CSV. "
        if failed_files:
            message += f"Note: Some files had issues or no instruments: {', '.join(failed_files)}. "
        message += "Your combined results are ready for download."

        return render_template(
            'index.html', 
            message=message, 
            output_file=output_csv_filename # Pass the name of the single CSV file for download
        )
    else:
        # If no CSVs were generated successfully (all failed or had no instruments)
        error_message = "No P&ID files were successfully processed or yielded instruments."
        if failed_files:
            error_message += f" Details: {', '.join(failed_files)}."
        return render_template('index.html', message=error_message, error=True)

    # This catch-all return should ideally not be reached if all cases are handled above
    return render_template('index.html', message='Invalid file type or no file selected.', error=True)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    # This route now serves the single combined CSV file
    return send_from_directory(
        app.config['UPLOAD_FOLDER'],
        filename,
        as_attachment=True,         # Forces download
        download_name=filename      # Specifies downloaded file name
    )

if __name__ == '__main__':
    app.run(debug=True)