import os
import cv2
import numpy as np
from flask import Flask, request, render_template, redirect, url_for, flash
from code2 import affine_correct 
from code2 import get_waist, get_chest, get_sleeve, get_shoulder_to_waist, get_shoulder_to_knee, get_shoulder_length # Ensure this function is defined in code2.py
import subprocess

app = Flask(__name__)
app.secret_key = 'your_secret_key'
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def get_waist(image, metre_pixel_x, metre_pixel_y):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        waist_pixels = cv2.arcLength(largest_contour, True)
        scale_factor = 0.3  # Decrease this value to reduce waist measurement
        waist_meters = waist_pixels * scale_factor * (metre_pixel_x + metre_pixel_y) / 2
       
        return waist_meters
    return 0.5  # Default value if no contours found

def get_chest(image, metre_pixel_x, metre_pixel_y):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        chest_pixels = cv2.arcLength(largest_contour, True)
        scale_factor = 2.0 # Example: Increase the scale factor
        chest_meters = chest_pixels * scale_factor * (metre_pixel_x + metre_pixel_y) / 2
        
        return chest_meters
    return 0.6  # Default value

def get_sleeve(image, metre_pixel_x, metre_pixel_y):
    height, _ = image.shape[:2]
    sleeve_length = height * 0.3  # Example: assuming sleeve length is 30% of image height
    return sleeve_length * (metre_pixel_x + metre_pixel_y) / 2

def get_shoulder_to_waist(image, metre_pixel_x, metre_pixel_y):
    height, _ = image.shape[:2]
    shoulder_to_waist = height * 0.6  # Example: 40% of image height
    return shoulder_to_waist * (metre_pixel_x + metre_pixel_y) / 2

def get_shoulder_to_knee(image, metre_pixel_x, metre_pixel_y):
    height, _ = image.shape[:2]
    shoulder_to_knee = height * 1.1 # Example: 60% of image height
    return shoulder_to_knee * (metre_pixel_x + metre_pixel_y) / 2

def get_shoulder_length(image, metre_pixel_x, metre_pixel_y):
    _, width = image.shape[:2]
    shoulder_length = width * 0.75  # Example: 15% of image width
    return shoulder_length * (metre_pixel_x + metre_pixel_y) / 2

def get_body_measurements(image):
    global metre_pixel_x, metre_pixel_y
    image = affine_correct(image)  # Correct the image if needed
    waist = get_waist(image, metre_pixel_x, metre_pixel_y)
    chest = get_chest(image, metre_pixel_x, metre_pixel_y)
    sleeve_length = get_sleeve(image, metre_pixel_x, metre_pixel_y)
    shoulder_to_waist = get_shoulder_to_waist(image, metre_pixel_x, metre_pixel_y)
    shoulder_to_knee = get_shoulder_to_knee(image, metre_pixel_x, metre_pixel_y)
    shoulder_length = get_shoulder_length(image, metre_pixel_x, metre_pixel_y)
    
    # Convert measurements from meters to inches
    meter_to_inches = 39.3701
    return {
        "waist": waist * meter_to_inches,
        "chest": chest * meter_to_inches,
        "sleeve_length": sleeve_length * meter_to_inches,
        "shoulder_to_waist": shoulder_to_waist * meter_to_inches,
        "shoulder_to_knee": shoulder_to_knee * meter_to_inches,
        "shoulder_length": shoulder_length * meter_to_inches
    }

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if file:
            # Save the file
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Process the image
            image = cv2.imread(file_path)
            if image is None:
                flash('Error loading image')
                return redirect(request.url)

            # Set the meter per pixel values here
            global metre_pixel_x, metre_pixel_y
            metre_pixel_x = 0.0008   # Example value; update this based on your measurements
            metre_pixel_y = 0.0008 # Example value; update this based on your measurements

            # Get body measurements
            measurements = get_body_measurements(image)

            return render_template('index.html', measurements=measurements, filename=filename)

    return render_template('index.html')

@app.route('/generate_tshirt', methods=['POST'])
def generate_tshirt():
    # Retrieve the measurements from the request
    measurements = request.form.to_dict()
    
    # Prepare the command to launch the Tkinter script
    command = ['python', 'tshirtgen.py']  # Changed to your Tkinter script name
    for key, value in measurements.items():
        command.append(f'{key}={value}')  # Pass the measurements as arguments

    # Launch the Tkinter app
    subprocess.Popen(command)

    return redirect(url_for('upload_file'))  # Redirect back to the upload page


if __name__ == "__main__":
    app.run(debug=True)
