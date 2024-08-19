from flask_cors import CORS, cross_origin
from flask import Flask, send_file
from PIL import Image
import flask
import numpy as np
import cv2
import os
import requests
import onnxruntime

# Define the URL and the local filename
url = "https://www.dropbox.com/scl/fi/64epu0r8x9opz6e9oav2i/sam_vit_b_encoder.onnx?rlkey=svsyckh7bijjyvyi0hlv808dr&st=bska00wy&dl=1"
local_filename = "sam_vit_b_encoder.onnx"

# Download the file if it doesn't exist locally
if not os.path.exists(local_filename):
    response = requests.get(url)
    with open(local_filename, 'wb') as f:
        f.write(response.content)

# Load the ONNX model using the local file path
encoder_session = onnxruntime.InferenceSession(local_filename)

app = Flask(__name__)
CORS(app, expose_headers=["Content-Disposition"])

@app.route('/', methods=['GET'])
@cross_origin(origin='http://localhost:3000')
def home():
    return "Hello, World !!"

@app.route('/test',methods=['POST'])
def testpost():
    imagfile = flask.request.files['image']
    print(imagfile)
    return "Image Received"

@app.route('/getembedding', methods=['POST'])
def getembedding1():
    try:
        # Check if the image file is present in the request
        if 'image' not in flask.request.files:
            return flask.jsonify({"error": "No image file provided in the request"}), 400

        imagefile = flask.request.files['image']

        # Check if the file has a valid image format
        try:
            img = Image.open(imagefile)
            img.verify()  # Verify that it is an image
        except (IOError, SyntaxError):
            return flask.jsonify({"error": "Invalid image file"}), 400

        # Convert image to numpy array
        cv_image = np.array(img)
        if cv_image.ndim != 3 or cv_image.shape[2] != 3:
            return flask.jsonify({"error": "Image must be a 3-channel RGB image"}), 400

        # Image resizing and transformation
        input_size = (684, 1024)
        scale_x = input_size[1] / cv_image.shape[1]
        scale_y = input_size[0] / cv_image.shape[0]
        scale = min(scale_x, scale_y)
        transform_matrix = np.array(
            [
                [scale, 0, 0],
                [0, scale, 0],
                [0, 0, 1],
            ]
        )
        cv_image = cv2.warpAffine(
            cv_image,
            transform_matrix[:2],
            (input_size[1], input_size[0]),
            flags=cv2.INTER_LINEAR,
        )

        # Prepare inputs for the ONNX model
        encoder_inputs = {
            "input_image": cv_image.astype(np.float32),
        }

        # Perform model inference
        try:
            output = encoder_session.run(None, encoder_inputs)
            if output is not None:
                output = output[0]
            image_embedding = output
        except Exception as e:
            return flask.jsonify({"error": "Error during model inference: " + str(e)}), 500

        return flask.jsonify(image_embedding.tolist())

    except Exception as e:
        return flask.jsonify({"error": "An unexpected error occurred: " + str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
