from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import keras

app = Flask(__name__)

model = keras.models.load_model('tumorsModel.h5')


def names(number):
    if number == 0:
        return 'Its a Tumor'
    else:
        return 'No, Its not a tumor'


@app.route('/predict', methods=['POST'])
def predict():
    if not request.files:
        return "No file upload", 400

    for file_name in request.files:
        file = request.files[file_name]

        if file and allowed_file(file.filename):
            img = Image.open(file.stream)
            x = np.array(img.resize((128, 128)))
            x = x.reshape(1, 128, 128, 3)
            res = model.predict_on_batch(x)
            classification = np.where(res == np.amax(res))[1][0]
            result = {
                'confidence': str(res[0][classification] * 100),
                'conclusion': names(classification)
            }
            return jsonify(result)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}


if __name__ == '__main__':
    app.run(debug=True)
