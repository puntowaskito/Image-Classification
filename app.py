import base64
import numpy as np
import io

from PIL import Image
from keras.applications.resnet import ResNet50, decode_predictions

from flask import Flask, request, jsonify


app = Flask(__name__)

IMG_SHAPE = (224, 224, 3)
def get_model():
    model = ResNet50(include_top=True, weights="imagenet", input_shape=IMG_SHAPE)
    print("[+] model loaded")
    return model

# decode the imaeg coming from the request
def decode_request(req):
    encoded = req["image"]
    decoded = base64.b64decode(encoded)
    return decoded

# preprocess image before sending it to the model
def preprocess(decoded):
    #resize and convert in RGB in case image is in RGBA
    pil_image = Image.open(io.BytesIO(decoded)).resize((224,224), Image.LANCZOS).convert("RGB")
    image = np.asarray(pil_image)
    batch = np.expand_dims(image, axis=0)

    return batch


model = get_model()

@app.route("/predict", methods=["POST"])
def predict():
    print("[+] request received")

    req = request.get_json(force=True)
    image = decode_request(req)
    batch = preprocess(image)

    prediction = model.predict(batch)

    p = decode_predictions(prediction)[0][0]
    top_label = [p[1], str(p[2])]

    response = {"prediction": top_label}
    print("[+] results {}".format(response))

    return jsonify(response)

if __name__ == "__main__":
    app.run(load_dotenv=True)