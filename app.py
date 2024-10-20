from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
from keras.models import load_model
import os

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Load model yang telah disimpan
model = load_model('apple_classifier.h5')

# Ukuran gambar yang sesuai dengan model
img_size = 224  # Sesuaikan dengan ukuran gambar yang dipakai di model
categories = ['20%', '40%', '60%', '80%', '100%']  # Sesuaikan dengan kategori klasifikasi apel

# Folder untuk menyimpan gambar yang diunggah
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Fungsi untuk prediksi gambar
def predict_image(img_path):
    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, (img_size, img_size))
    img_normalized = img_resized / 255.0
    img_reshaped = np.reshape(img_normalized, (1, img_size, img_size, 3))

    prediction = model.predict(img_reshaped)
    class_idx = np.argmax(prediction)
    return categories[class_idx]

# Halaman utama untuk mengunggah gambar
@app.route('/')
def index():
    return render_template('index.html')

# Proses unggah dan prediksi gambar
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Prediksi gambar
        predicted_class = predict_image(filepath)

        return render_template('index.html', prediction=predicted_class, img_path=filepath)

if __name__ == '__main__':
    app.run(debug=True)
