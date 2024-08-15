from flask_cors import CORS, cross_origin
from flask import Flask, send_file
from PIL import Image
import flask
import numpy as np
import cv2
import os
import onnxruntime



encoder_path = "https://www.dropbox.com/scl/fi/64epu0r8x9opz6e9oav2i/sam_vit_b_encoder.onnx?rlkey=svsyckh7bijjyvyi0hlv808dr&st=bska00wy&dl=1"
encoder_session = onnxruntime.InferenceSession(
    encoder_path, providers=['CPUExecutionProvider'])
app = Flask(__name__)
CORS(app, expose_headers=["Content-Disposition"])


@app.route('/', methods=['GET'])
@cross_origin(origin='http://localhost:3000')
def home():
    return "Hello, World !!"

@app.route('/test',methods=['POST'])
def testpost():
    imagfile=flask.request.files['image']
    print(imagfile)
    return "Image Recieved"

@app.route('/getembedding', methods=['POST'])
# @cross_origin(origin='http://localhost:3000')
def getembedding1():
    imagefile = flask.request.files['image']
    print(imagefile)
    img = Image.open(imagefile)
    cv_image = np.array(img)
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
    encoder_inputs = {
        "input_image": cv_image.astype(np.float32),
    }
    output = encoder_session.run(None, encoder_inputs)
    if output is not None:
        output = output[0]
    image_embedding = output
    return flask.jsonify(image_embedding.tolist())


if (__name__ == '__main__'):
    app.run(debug=True,port=os.getenv("PORT", default=10000),host=os.getenv("HOST", default="0.0.0.0"))
