from flask import Flask, request, jsonify
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

try:
    language_model = load_model('./language_identifier.h5')
    era_model = load_model('./era_predict_model.h5')
    print("Models loaded successfully")
except Exception as e:
    print(f"Error loading models: {str(e)}")

languages = ['pali', 'sanskrutha', 'sinhala']
era = ['mahanuwara', 'polonnaruwa']

def predict_image_class(image_path, model, class_names):
    try:
        img = load_img(image_path, target_size=(180, 180, 3))  
        img_array = img_to_array(img)

        img_array = np.expand_dims(img_array, axis=0)  

        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions)]
        return predicted_class
    except Exception as e:
        print(f"Error in image processing: {str(e)}")
        return None, str(e)

@app.route('/', methods=['GET'])
def hello_world():
    print("Hello World!")  
    return jsonify({"message": "Hello, World!"}), 200

@app.route('/upload-image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        print("No image part in the request")  
        return jsonify({"error": "No image part in the request"}), 400

    image = request.files['image']

    if image.filename == '':
        print("No file selected")  
        return jsonify({"error": "No file selected"}), 400

    if image:
        try:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
            image.save(image_path)

            language_predict = predict_image_class(image_path, language_model, languages)
            era_predict = predict_image_class(image_path, era_model, era)

            if language_predict is None:
                return jsonify({"error": "Prediction failed"}), 500

            print(f"Predicted Language - {language_predict}")
            print(f"Predicted Era - {era_predict}")

            return jsonify({
                "message": "Image uploaded and prediction successful",
                "filename": image.filename,
                "path": image_path,
                "language": language_predict,
                "era": era_predict
            }), 200
        except Exception as e:
            print(f"Error: {str(e)}")  
            return jsonify({"error": "File upload or prediction failed", "details": str(e)}), 500

    print("File upload failed")
    return jsonify({"error": "File upload failed"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
