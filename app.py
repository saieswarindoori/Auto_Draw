import os
import pandas as pd
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from werkzeug.utils import secure_filename
from instrument_processor import InstrumentProcessor
from config import (
    UPLOAD_FOLDER, POPPLER_PATH, GOOGLE_APPLICATION_CREDENTIALS_PATH,
    DEBUG_MODE, DEBUG_OUTPUT_FOLDER
)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'your_secret_key_here' # You should use a strong, random secret key in production

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Initialize the InstrumentProcessor (without legend data initially)
processor = InstrumentProcessor()

# Helper function to check allowed extensions
def allowed_file(filename, allowed_extensions):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    pid_file = request.files.get('pid_file')
    legend_file = request.files.get('legend_file')
    
    # Check if a PID file was provided
    if not pid_file or pid_file.filename == '':
        flash('No P&ID file selected.', 'danger')
        return redirect(url_for('index'))

    # Check allowed extensions for P&ID file
    if not allowed_file(pid_file.filename, {'pdf'}):
        flash('Invalid P&ID file type. Only PDF is allowed.', 'danger')
        return redirect(url_for('index'))

    pid_filename = secure_filename(pid_file.filename)
    pid_path = os.path.join(app.config['UPLOAD_FOLDER'], pid_filename)
    pid_file.save(pid_path)
    print(f"P&ID file saved to: {pid_path}")
    flash(f'P&ID file "{pid_filename}" uploaded successfully.', 'success')

    # Handle legend file if provided
    if legend_file and legend_file.filename != '':
        legend_filename = secure_filename(legend_file.filename)
        # Check allowed extensions for legend file
        if not allowed_file(legend_filename, {'csv', 'xls', 'xlsx'}):
            flash('Invalid legend file type. Only CSV or Excel (xls, xlsx) are allowed.', 'danger')
            return redirect(url_for('index'))

        legend_path = os.path.join(app.config['UPLOAD_FOLDER'], legend_filename)
        legend_file.save(legend_path)
        print(f"Legend file saved to: {legend_path}")

        # Initialize legend_df here to ensure it always exists in this scope
        legend_df = pd.DataFrame() 

        try:
            # Determine file type and load accordingly
            if legend_filename.endswith('.csv'):
                legend_df = pd.read_csv(legend_path)
            elif legend_filename.endswith(('.xls', '.xlsx')):
                legend_df = pd.read_excel(legend_path)
            # No 'else' needed here, as allowed_file check already handles unsupported types.
            # If the file extension is valid but it's not a proper CSV/Excel,
            # pd.read_csv/read_excel will raise an exception, caught by the outer 'except'.
            
            processor.load_legend_data(legend_df) 
            flash(f'Legend file "{legend_filename}" processed successfully.', 'info')

        except Exception as e:
            flash(f'Error processing legend file: {e}', 'danger')
            print(f"Error processing legend file: {e}")
            return redirect(url_for('index'))
    else:
        # If no legend file is provided, clear any previously loaded legend data
        processor.load_legend_data(pd.DataFrame()) 
        flash('No legend file provided. Instrument tags will not be filtered by type.', 'warning')


    # Process the P&ID file
    try:
        filename_base = os.path.splitext(pid_filename)[0]
        final_instruments_df = processor.process_pid_file(pid_path, filename_base)

        if not final_instruments_df.empty:
            output_csv_filename = f"{filename_base}_final_instruments.csv"
            output_csv_path = os.path.join(app.config['UPLOAD_FOLDER'], output_csv_filename)
            final_instruments_df.to_csv(output_csv_path, index=False)
            flash(f'Instruments extracted successfully! Results saved to: {output_csv_filename}', 'success')
            
            # Optionally, provide a link to download the CSV if Flask is set up for serving static files
            # For local debugging, just printing the path is fine.
            print(f"Final instruments saved to: {output_csv_path}")
            
            # Prepare data for display (e.g., first few rows) or for further action
            return render_template(
                'index.html', 
                table=final_instruments_df.head().to_html(classes='data'), 
                titles=final_instruments_df.columns.values
            )
        else:
            flash('No instruments were extracted from the P&ID with the current settings. Try adjusting parameters.', 'warning')
            return redirect(url_for('index'))

    except Exception as e:
        flash(f'An error occurred during processing: {e}', 'danger')
        print(f"An error occurred during processing: {e}")
        import traceback
        traceback.print_exc() # Print full traceback to console for debugging
        return redirect(url_for('index'))

if __name__ == '__main__':
    # Set debug mode for Flask
    app.debug = DEBUG_MODE 
    app.run(debug=DEBUG_MODE) # Run in debug mode if DEBUG_MODE is True in config