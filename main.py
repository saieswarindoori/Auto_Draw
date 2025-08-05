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


from concurrent.futures import ThreadPoolExecutor, as_completed

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

    # ---- Process Legend File ----
    legend_df = pd.DataFrame()
    if legend_file and legend_file.filename != '':
        legend_filename = secure_filename(legend_file.filename)
        legend_file_path = os.path.join(app.config['UPLOAD_FOLDER'], legend_filename)
        legend_file.save(legend_file_path)

        if legend_filename.lower().endswith(('.xlsx', '.xls')):
            legend_df = pd.read_excel(legend_file_path)
        elif legend_filename.lower().endswith('.csv'):
            legend_df = pd.read_csv(legend_file_path)
        
        if not legend_df.empty:
            legend_df.columns = [col.strip() for col in legend_df.columns]
            if 'Instrument Type' in legend_df.columns:
                legend_df = legend_df.rename(columns={'Instrument Type': 'Instrument_Type'})

        processor.load_legend_data(legend_df)
    else:
        processor.load_legend_data(pd.DataFrame())

    # ---- Parallel Processing P&ID Files ----
    all_instruments_data = []
    processed_count = 0
    failed_files = []

    def process_single_pid(pid_file):
        try:
            pid_filename = secure_filename(pid_file.filename)
            pid_file_path = os.path.join(app.config['UPLOAD_FOLDER'], pid_filename)
            pid_file.save(pid_file_path)

            filename_base = os.path.splitext(pid_filename)[0]
            final_instruments_df = processor.process_pid_file(pid_file_path, filename_base)

            if not final_instruments_df.empty:
                final_instruments_df['Source_PID_File'] = pid_filename

                if 'Y_Coordinate' in final_instruments_df.columns and 'X_Coordinate' in final_instruments_df.columns:
                    final_instruments_df = final_instruments_df.sort_values(
                        by=['Y_Coordinate', 'X_Coordinate'], ascending=[True, True]
                    ).reset_index(drop=True)

                return final_instruments_df, None
            else:
                return None, f"{pid_filename} (no instruments detected)"
        except Exception as e:
            return None, f"{pid_file.filename} (Error: {e})"
        finally:
            if os.path.exists(pid_file_path):
                os.remove(pid_file_path)

    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_file = {
            executor.submit(process_single_pid, pid_file): pid_file.filename
            for pid_file in pid_files if pid_file and allowed_file(pid_file.filename)
        }

        for future in as_completed(future_to_file):
            result_df, error = future.result()
            if result_df is not None:
                all_instruments_data.append(result_df)
                processed_count += 1
            else:
                failed_files.append(error)

    # ---- Combine and Export Data ----
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

            # Reorder columns
            cols = combined_df.columns.tolist()
            if 'Description' in cols and 'Instrument_Type' in cols:
                desc_idx = cols.index('Description')
                type_idx = cols.index('Instrument_Type')
                if desc_idx != type_idx + 1:
                    cols.insert(type_idx + 1, cols.pop(desc_idx))
                    combined_df = combined_df[cols]

        output_excel_filename = "combined_pid_instruments.xlsx"
        output_excel_path = os.path.join(app.config['UPLOAD_FOLDER'], output_excel_filename)
        combined_df.to_excel(output_excel_path, index=False)

        message = f"Successfully processed {processed_count} P&ID file(s)."
        if failed_files:
            message += f" Some files had issues: {', '.join(failed_files)}."

        debug_image_filename = f"{os.path.splitext(pid_files[0].filename)[0]}_circles_detected.jpg"

        return render_template(
            'index.html',
            message=message,
            output_file=output_excel_filename,
            debug_image_link=url_for('download_debug_image', filename=debug_image_filename)
        )

    else:
        error_message = "No P&ID files were successfully processed."
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
@app.route('/debug/<filename>')
def download_debug_image(filename):
    """Serves the debug circles detected images for download."""
    return send_from_directory(
        DEBUG_OUTPUT_FOLDER,
        filename,
        as_attachment=True
    )

if __name__ == '__main__':
    # Run the Flask development server
    app.run(debug=True)