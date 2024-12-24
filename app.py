import os
import psycopg2
from flask import Flask, render_template, request, send_from_directory
import cv2
import numpy as np
import requests
import sys
import json
from datetime import datetime
import boto3
from botocore.exceptions import NoCredentialsError
from dotenv import load_dotenv


load_dotenv()

# Настройка приложения Flask
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
OUTPUT_FOLDER = 'static/output/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Настройки Object Storage
S3_ENDPOINT_URL = os.getenv('S3_ENDPOINT_URL')
S3_ACCESS_KEY = os.getenv('S3_ACCESS_KEY')
S3_SECRET_KEY = os.getenv('S3_SECRET_KEY')
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME')

# Клиент S3
s3_client = boto3.client(
    's3',
    endpoint_url=S3_ENDPOINT_URL,
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY
)

# Данные для подключения к базе данных
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')

# SmartCaptcha Server Key
SMARTCAPTCHA_SERVER_KEY = os.getenv('SMARTCAPTCHA_SERVER_KEY')

def upload_to_s3(file_path, filename): # Сохранение картинки в data storage
    try:
        s3_client.upload_file(file_path, S3_BUCKET_NAME, filename)
        print(f"Файл {filename} успешно загружен в бакет {S3_BUCKET_NAME}.")
        return True
    except FileNotFoundError:
        print(f"Файл {file_path} не найден.")
        return False
    except NoCredentialsError:
        print("Ошибка доступа: неверные ключи доступа.")
        return False

# Подключение к базе данных
def connect_db():
    try:
        return psycopg2.connect(f"""
            host={DB_HOST}
            port={DB_PORT}
            sslmode=verify-full
            dbname={DB_NAME}
            user={DB_USER}
            password={DB_PASSWORD}
            target_session_attrs=read-write
    """)
    except Exception as e:
        print(f'Ошибка подключения к базе данных: {e}')

# Создание таблицы (один раз при запуске приложения)
def create_table():
    try:
        conn = connect_db()
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS photo_logs (
                id SERIAL PRIMARY KEY,
                filename TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL
            );
        """)
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Ошибка создания таблицы: {e}", file=sys.stderr)

# Загрузка конфигурации и весов модели MobileNet SSD
prototxt_path = 'deploy.prototxt'
caffe_model_path = 'mobilenet_iter_73000.caffemodel'

# Загрузка списка классов
class_names = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"
]

# Загрузка модели
net = cv2.dnn.readNetFromCaffe(prototxt_path, caffe_model_path)

def check_captcha(token):
    resp = requests.post(
        "https://smartcaptcha.yandexcloud.net/validate",
        data={
            "secret": SMARTCAPTCHA_SERVER_KEY,
            "token": token,
            # "ip": "127.0.0.1"
            "ip": "89.169.169.188"
        },
        timeout=1
    )
    server_output = resp.content.decode()
    print(f'server_output: {server_output}')
    if resp.status_code != 200:
        print(f"Allow access due to an error: code={resp.status_code}; message={server_output}", file=sys.stderr)
        return True
    return json.loads(server_output)["status"] == "ok"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file and check_captcha(request.form["smart-token"]):
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Загрузка файла в Object Storage
        if upload_to_s3(file_path, filename):
            print(f"Файл {filename} загружен в Object Storage.")


        # Обработка изображения с помощью нейронной сети
        image = cv2.imread(file_path)
        (h, w) = image.shape[:2]

        # Подготовка изображения для модели (масштабирование)
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.2:
                class_id = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                label = f"{class_names[class_id]}: {confidence * 100:.2f}%"
                cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(image, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        output_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
        cv2.imwrite(output_path, image)

        # Сохранение данных в базу
        try:
            conn = connect_db()
            cursor = conn.cursor()
            timestamp = datetime.now()
            cursor.execute(
                "INSERT INTO photo_logs (filename, timestamp) VALUES (%s, %s)",
                (filename, timestamp)
            )
            conn.commit()
            cursor.close()
            conn.close()
        except Exception as e:
            print(f"Error inserting data: {e}", file=sys.stderr)

        return render_template('index.html', uploaded_image=file_path, output_image=output_path)
    else:
        return render_template('index.html')

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

create_table()
app.run(host='0.0.0.0', port=8080, debug=True)
