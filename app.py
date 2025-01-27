import os
import cv2
from flask import Flask, render_template, Response
from inference import get_model
import supervision as sv
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.cluster import DBSCAN
from threading import Lock
import logging
import time
import openpyxl

# Initialize Flask app
app = Flask(__name__)

# Configuration
API_KEY = os.getenv("ROBOFLOW_API_KEY", "8ocDcITWfcxr4vnt5oEz")
MODEL_ID = "ocr-uikzj/5"
OUTPUT_DIR = "data_output"
EXCEL_FILE = os.path.join(OUTPUT_DIR, "multimeter_values.xlsx")
CACHE_DURATION = 60  # Cache duration in seconds

# Initialize Roboflow model
model = get_model(model_id=MODEL_ID, api_key=API_KEY)

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(filename="app.log", level=logging.DEBUG, 
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize CSV file with headers
def initialize_data_file():
    try:
        if not os.path.exists(EXCEL_FILE):
            df = pd.DataFrame(columns=["Timestamp", "Multimeter 1", "Multimeter 2", "Multimeter 3"])
            df.to_excel(EXCEL_FILE, index=False, engine='openpyxl')
            logging.info(f"Data file created: {EXCEL_FILE}")
    except Exception as e:
        logging.error(f"Error initializing data file: {e}")

initialize_data_file()

# Thread-safe lock for file operations
file_lock = Lock()

# Track last recorded values for multimeters
last_values = {
    "Multimeter 1": {"value": None, "timestamp": time.time()},
    "Multimeter 2": {"value": None, "timestamp": time.time()},
    "Multimeter 3": {"value": None, "timestamp": time.time()},
}

def preprocess_image(image):
    """Preprocesses an image for OCR."""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    except Exception as e:
        logging.error(f"Error in preprocess_image: {e}")
        return image

def validate_and_format_value(class_ids):
    """Validates and formats OCR-detected values."""
    # Mapping of detected classes to digits
    digit_map = {
        "0": "0", "1": "1", "2": "2", "3": "3", "4": "4", 
        "5": "5", "6": "6", "7": "7", "8": "8", "9": "9", 
        "dot": "."
    }
    
    # Filter and map detected classes
    valid_digits = [digit_map.get(cls, '') for cls in class_ids if cls in digit_map]
    
    logging.debug(f"Valid digits before combining: {valid_digits}")
    
    # Combine digits
    value = ''.join(valid_digits)
    
    logging.debug(f"Combined value before validation: {value}")
    
    # Validate value
    try:
        float_val = float(value) if value else 0.0
        # Ensure value is between 0 and 1000
        validated_value = str(max(0, min(float_val, 1000)))
        logging.debug(f"Final validated value: {validated_value}")
        return validated_value
    except ValueError:
        logging.debug("Value validation failed, returning 0.0")
        return "0.0"

def save_to_data_file(timestamp, multimeter_data):
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(EXCEL_FILE), exist_ok=True)
        
        # Check if any value is non-zero
        non_zero_values = [float(value) for value in multimeter_data.values() if float(value) > 0]
        
        if not non_zero_values:
            logging.debug("No non-zero values to save")
            return
        
        # Read existing data or create new DataFrame
        try:
            df = pd.read_excel(EXCEL_FILE, engine='openpyxl')
        except Exception as e:
            logging.warning(f"Cannot read existing file, creating new: {e}")
            df = pd.DataFrame(columns=["Timestamp", "Multimeter 1", "Multimeter 2", "Multimeter 3"])
        
        # Create new row
        new_row = pd.DataFrame({
            'Timestamp': [timestamp],
            'Multimeter 1': [multimeter_data.get('Multimeter 1', '0.0')],
            'Multimeter 2': [multimeter_data.get('Multimeter 2', '0.0')],
            'Multimeter 3': [multimeter_data.get('Multimeter 3', '0.0')]
        })
        
        # Concatenate and save
        df = pd.concat([df, new_row], ignore_index=True)
        
        # Ensure file is writable
        if os.path.exists(EXCEL_FILE):
            os.chmod(EXCEL_FILE, 0o666)
        
        # Save with error handling
        try:
            df.to_excel(EXCEL_FILE, index=False, engine='openpyxl')
            logging.info(f"Data saved: {new_row}")
        except PermissionError:
            logging.error(f"Permission denied when writing to {EXCEL_FILE}")
        except Exception as e:
            logging.error(f"Error saving Excel file: {e}")
    
    except Exception as e:
        logging.error(f"Unexpected error in save_to_data_file: {e}")
        import traceback
        logging.error(traceback.format_exc())

def gen_frames():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    while True:
        ret, frame = cap.read()
        if not ret:
            logging.error("Failed to capture frame from webcam.")
            break

        try:
            processed_frame = preprocess_image(frame)
            results = model.infer(processed_frame)[0]
            predictions = results.predictions

            # Log total number of predictions
            logging.debug(f"Total predictions: {len(predictions)}")
            logging.debug(f"Prediction details: {[(p.class_name, p.x, p.y) for p in predictions]}")

            # Sort predictions by x-coordinate
            sorted_predictions = sorted(predictions, key=lambda pred: pred.x)
            
            # Prepare multimeter data dictionary
            multimeter_data = {
                "Multimeter 1": "0.0", 
                "Multimeter 2": "0.0", 
                "Multimeter 3": "0.0"
            }

            # Limit processing to first 3 detections
            for i, pred in enumerate(sorted_predictions[:3]):
                # Skip 'dot' class
                if pred.class_name == 'dot':
                    continue

                # Crop region of interest
                x, y, w, h = int(pred.x), int(pred.y), int(pred.width), int(pred.height)
                cropped = frame[max(0, y):y+h, max(0, x):x+w]
                
                # Preprocess cropped image
                cropped_processed = preprocess_image(cropped)
                
                # Perform OCR on cropped region
                ocr_results = model.infer(cropped_processed)[0]
                
                # Extract class names
                detected_class_ids = [p.class_name for p in ocr_results.predictions]
                logging.debug(f"OCR for Multimeter {i+1} - Raw class IDs: {detected_class_ids}")
                
                # Validate and format value
                detected_value = validate_and_format_value(detected_class_ids)
                logging.debug(f"Multimeter {i+1} - Validated value: {detected_value}")
                
                # Map to specific multimeter column based on detection order
                multimeter_data[f"Multimeter {i+1}"] = detected_value

            # Log and save multimeter data
            logging.debug(f"Multimeter data: {multimeter_data}")
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            save_to_data_file(timestamp, multimeter_data)

            # Annotate frame
            detections = sv.Detections.from_inference(results)
            annotated_frame = box_annotator.annotate(scene=frame, detections=detections)
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)
            
            # Encode and yield frame
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        except Exception as e:
            logging.error(f"Error during processing: {e}")
            break

    cap.release()

# Routes remain the same
@app.route("/")
def index():
    return render_template("display.html")

@app.route("/webcam_feed")
def webcam_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")
@app.route("/home")
def home():
    """Render the homepage (index.html)."""
    return render_template("index.html")

@app.route('/daywise_report')
def daywise_report():
    return render_template('daywise_report.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)