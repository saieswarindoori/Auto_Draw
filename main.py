import os
import pandas as pd
from flask import Flask, request, render_template, send_from_directory, jsonify, url_for
from werkzeug.utils import secure_filename
from instrument_processor import InstrumentProcessor
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import configuration settings
from config import (
    UPLOAD_FOLDER, ALLOWED_EXTENSIONS, GOOGLE_APPLICATION_CREDENTIALS_PATH, POPPLER_PATH,
    PDF_DPI, HOUGH_DP, HOUGH_MIN_DIST, HOUGH_PARAM1, HOUGH_PARAM2,
    HOUGH_MIN_RADIUS, HOUGH_MAX_RADIUS,
    OCR_ROI_MARGIN_FACTOR, TEXT_CONCAT_SEPARATOR,
    DEBUG_MODE, DEBUG_OUTPUT_FOLDER,
    ANCHOR_MIN_COUNT, RADIUS_TOLERANCE_PERCENT
)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload size

# Ensure upload and debug folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DEBUG_OUTPUT_FOLDER, exist_ok=True)

# Initialize the InstrumentProcessor (only once per Flask app instance)
processor = InstrumentProcessor()

def allowed_file(filename, allowed_extensions):
    """Checks if a file's extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions

@app.route('/')
def index():
    """Renders the main upload page."""
    # This route now simply serves the initial HTML.
    # All dynamic content (processing, results) is handled via AJAX.
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    print("Received upload request.") # Debug print

    pid_files = request.files.getlist('pid_file')
    legend_file = request.files.get('legend_file')

    if not pid_files or all(f.filename == '' for f in pid_files):
        print("No P&ID files selected.") # Debug print
        return jsonify(message='No P&ID files selected.', error=True), 400

    if len(pid_files) > 50:
        print("Too many P&ID files selected.") # Debug print
        return jsonify(message='Maximum 50 P&ID files allowed per upload.', error=True), 400

    # ---- Process Legend File ----
    legend_df = pd.DataFrame()
    if legend_file and legend_file.filename != '':
        legend_filename = secure_filename(legend_file.filename)
        legend_file_path = os.path.join(app.config['UPLOAD_FOLDER'], legend_filename)
        legend_file.save(legend_file_path)

        try:
            if legend_filename.lower().endswith(('.xlsx', '.xls')):
                legend_df = pd.read_excel(legend_file_path)
            elif legend_filename.lower().endswith('.csv'):
                legend_df = pd.read_csv(legend_file_path)
            
            if not legend_df.empty:
                legend_df.columns = [col.strip() for col in legend_df.columns]
                if 'Instrument Type' in legend_df.columns:
                    legend_df = legend_df.rename(columns={'Instrument Type': 'Instrument_Type'})
                print("Legend data loaded successfully.") # Debug print
        except Exception as e:
            print(f"Error loading legend file: {e}") # Debug print
            # Continue without legend if there's an error
        finally:
            if os.path.exists(legend_file_path):
                os.remove(legend_file_path) # Clean up legend file after processing
    else:
        print("No legend file provided or invalid format. Proceeding without legend data.") # Debug print
    
    processor.load_legend_data(legend_df) # Always load, even if empty

    # ---- Parallel Processing P&ID Files ----
    all_instruments_data = []
    processed_count = 0
    failed_files = []
    
    # Store original file objects to delete temporary files later
    temp_pid_files_to_delete = []

    def process_single_pid(pid_file_obj):
        try:
            pid_filename = secure_filename(pid_file_obj.filename)
            pid_file_path = os.path.join(app.config['UPLOAD_FOLDER'], pid_filename)
            pid_file_obj.save(pid_file_path)
            temp_pid_files_to_delete.append(pid_file_path) # Mark for deletion
            print(f"Saved temporary P&ID file: {pid_file_path}") # Debug print

            filename_base = os.path.splitext(pid_filename)[0]
            # CORRECTED: Changed process_pid to process_pid_file
            final_instruments_df = processor.process_pid_file(pid_file_path, filename_base)

            if not final_instruments_df.empty:
                final_instruments_df['Source_PID_File'] = pid_filename

                # Ensure X_Coordinate and Y_Coordinate exist before sorting
                if 'X_Coordinate' in final_instruments_df.columns and 'Y_Coordinate' in final_instruments_df.columns:
                    final_instruments_df = final_instruments_df.sort_values(
                        by=['Y_Coordinate', 'X_Coordinate'], ascending=[True, True]
                    ).reset_index(drop=True)

                print(f"Successfully processed P&ID: {pid_filename}") # Debug print
                return final_instruments_df, None
            else:
                print(f"No instruments extracted from {pid_filename}.") # Debug print
                return None, f"{pid_filename} (no instruments detected)"
        except Exception as e:
            print(f"Error processing {pid_filename}: {e}") # Debug print
            import traceback
            traceback.print_exc()
            return None, f"{pid_filename} (Error: {e})"

    with ThreadPoolExecutor(max_workers=4) as executor:
        # Filter out invalid files before submitting to executor
        valid_pid_files = [f for f in pid_files if f and allowed_file(f.filename, {'pdf'})]
        
        future_to_file = {
            executor.submit(process_single_pid, pid_file): pid_file.filename
            for pid_file in valid_pid_files
        }

        for future in as_completed(future_to_file):
            original_filename = future_to_file[future]
            try:
                result_df, error = future.result()
                if result_df is not None:
                    all_instruments_data.append(result_df)
                    processed_count += 1
                else:
                    failed_files.append(error)
            except Exception as exc:
                print(f'{original_filename} generated an exception: {exc}')
                failed_files.append(f"{original_filename} (Exception: {exc})")

    # Clean up temporary PID files
    for temp_file_path in temp_pid_files_to_delete:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            print(f"Cleaned up temporary file: {temp_file_path}") # Debug print

    # ---- Combine and Export Data (JSON Response) ----
    if all_instruments_data:
        combined_df = pd.concat(all_instruments_data, ignore_index=True)

        if not legend_df.empty and 'Instrument_Type' in legend_df.columns and 'Description' in legend_df.columns:
            legend_for_merge = legend_df[['Instrument_Type', 'Description']].drop_duplicates()

            if 'Description' in combined_df.columns:
                combined_df = combined_df.drop(columns=['Description'])

            combined_df = pd.merge(
                combined_df,
                legend_for_merge,
                on='Instrument_Type',
                how='left'
            )

            # Reorder columns to place Description next to Instrument_Type
            cols = combined_df.columns.tolist()
            if 'Description' in cols and 'Instrument_Type' in cols:
                desc_idx = cols.index('Description')
                type_idx = cols.index('Instrument_Type')
                if desc_idx != type_idx + 1: # Only move if not already adjacent
                    cols.insert(type_idx + 1, cols.pop(desc_idx))
                    combined_df = combined_df[cols]

        output_excel_filename = f"combined_pid_instruments_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        output_excel_path = os.path.join(app.config['UPLOAD_FOLDER'], output_excel_filename)
        combined_df.to_excel(output_excel_path, index=False)
        print(f"Final instruments saved to: {output_excel_path}") # Debug print

        message = f"Successfully processed {processed_count} P&ID file(s)."
        if failed_files:
            message += f" Some files had issues: {', '.join(failed_files)}."

        # Assuming the processor generates page1.jpg for the first processed PID
        first_pid_filename_base = os.path.splitext(secure_filename(pid_files[0].filename))[0] if pid_files else "default"
        debug_image_filename = f"{first_pid_filename_base}_circles_detected.jpg"
        debug_image_link = url_for('download_debug_image', filename=debug_image_filename)
        print(f"Debug image link: {debug_image_link}") # Debug print

        # Return JSON response for AJAX call
        return jsonify(
            message=message,
            output_file=output_excel_filename,
            debug_image_link=debug_image_link,
            table=combined_df.head().to_html(classes='data table-auto w-full text-left whitespace-no-wrap')
        ), 200

    else:
        error_message = "No P&ID files were successfully processed."
        if failed_files:
            error_message += f" Details: Failed to process: {', '.join(failed_files)}."
        print(f"Processing failed: {error_message}") # Debug print
        return jsonify(message=error_message, error=True), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serves the uploaded/generated files for download."""
    print(f"Serving uploaded file: {filename}") # Debug print
    return send_from_directory(
        app.config['UPLOAD_FOLDER'],
        filename,
        as_attachment=True,         # Force download
        download_name=filename      # Suggest original filename for download
    )

@app.route('/debug/<filename>')
def download_debug_image(filename):
    """Serves the debug circles detected images for download."""
    print(f"Serving debug image: {filename}") # Debug print
    # Ensure the file exists before sending
    file_path = os.path.join(DEBUG_OUTPUT_FOLDER, filename)
    if os.path.exists(file_path):
        return send_from_directory(
            DEBUG_OUTPUT_FOLDER,
            filename,
            as_attachment=True # Force download for debug images too
        )
    else:
        print(f"Debug image not found: {file_path}")
        # Return a 404 or a placeholder if the image doesn't exist
        return jsonify(message=f"Debug image '{filename}' not found.", error=True), 404


if __name__ == '__main__':
    # Run the Flask development server
    app.run(host='0.0.0.0', debug=DEBUG_MODE)

