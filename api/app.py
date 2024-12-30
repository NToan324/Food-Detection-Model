import os
import gdown
from flask import Flask, render_template, request, jsonify  
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
import tensorflow as tf
from flask_cors import CORS

app = Flask(__name__)

allowed_origins = ["https://food-detection.vercel.app", "http://localhost:3000", "https://food-detection.vercel.app"]
CORS(app, resources={r"/*": {"origins": allowed_origins}}, supports_credentials=True)

# Đường dẫn model trên Google Drive
MODEL_URL = "https://drive.google.com/uc?id=1LnUAXj3e1AlpYhO-UBJxx-Ugy39FC9u7"
MODEL_PATH = "../Model/ResNet50_v2.keras"

# Tải model nếu chưa tồn tại
if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

@app.route('/', methods=['GET'])    
def hello():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    model = tf.keras.models.load_model(MODEL_PATH)
    class_index = {
        'Bánh bèo': 0, 'Bánh bột lọc': 1, 'Bánh căn': 2, 'Bánh canh': 3, 
        'Bánh chưng': 4, 'Bánh cuốn': 5, 'Bánh đúc': 6, 'Bánh giò': 7, 
        'Bánh khọt': 8, 'Bánh mì': 9, 'Bánh pía': 10, 
        'Bánh tét': 11, 'Bánh tráng nướng': 12, 'Bánh xèo': 13, 'Bún bò Huế': 14, 
        'Bún đậu mắm tôm': 15, 'Bún mắm': 16, 'Bún riêu': 17, 'Bún thịt nướng': 18, 
        'Cá kho tộ': 19, 'Canh chua': 20, 'Cao lầu': 21, 'Cháo lòng': 22, 
        'Cơm tấm': 23, 'Gỏi cuốn': 24, 'Hủ tiếu': 25, 'Mì Quảng': 26, 
        'Nem chua': 27, 'Phở': 28,'Không tìm thấy': 29, 'Xôi xéo': 30 
    }

    try:
        # Nhận file từ request
        imagefile = request.files['imagefile']
        image_path = "../images/" + imagefile.filename
        imagefile.save(image_path)

        # Xử lý ảnh
        img = load_img(image_path, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Dự đoán
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions)
        result_predict = list(class_index.keys())[predicted_class]
        
        print(result_predict)

        # Trả dữ liệu JSON
        return jsonify({
            "success": True,
            "prediction": {"label": result_predict, "index": int(predicted_class)}
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })

if __name__ == '__main__':
    app.run(port=3001, debug=True)
