from ultralytics import YOLO
import cv2
import logging
import sqlite3
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify
import pandas as pd
import plotly.express as px
import threading

# Flask app setup
app = Flask(__name__)

def get_data():
    conn = sqlite3.connect('people_count.db')
    query = "SELECT timestamp, num_people FROM PeopleCount"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

@app.route('/')
def index():
    df = get_data()
    fig = px.line(df, x='timestamp', y='num_people', title='People Count Over Time')
    graph_html = fig.to_html(full_html=False)
    return render_template('index.html', graph_html=graph_html)

@app.route('/data')
def data():
    df = get_data()
    fig = px.line(df, x='timestamp', y='num_people', title='People Count Over Time')
    graph_html = fig.to_html(full_html=False)
    return jsonify(graph_html=graph_html)

def run_flask():
    app.run(debug=True, use_reloader=False)

# Start the Flask app in a separate thread
flask_thread = threading.Thread(target=run_flask)
flask_thread.start()

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Database setup
def create_database():
    conn = sqlite3.connect('people_count.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS PeopleCount (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            num_people INTEGER NOT NULL
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS PeopleTimer (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id INTEGER NOT NULL,
            entry_time TEXT NOT NULL,
            duration REAL NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def save_people_count(num_people):
    conn = sqlite3.connect('people_count.db')
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('''
        INSERT INTO PeopleCount (timestamp, num_people)
        VALUES (?, ?)
    ''', (timestamp, num_people))
    conn.commit()
    conn.close()

def save_person_timer(person_id, entry_time, duration):
    conn = sqlite3.connect('people_count.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO PeopleTimer (person_id, entry_time, duration)
        VALUES (?, ?, ?)
    ''', (person_id, entry_time.strftime("%Y-%m-%d %H:%M:%S"), duration))
    conn.commit()
    conn.close()

create_database()

# Load the YOLO model
model = YOLO('yolo11n.pt')

# RTSP stream URL
rtsp_url = "rtsp://admin1:admin1@192.168.0.100:554/stream1"

# Open the RTSP stream
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("Cannot open RTSP stream. Check the URL and connection.")
    exit()

frame_counter = 0
process_every_nth_frame = 5
last_saved_time = datetime.now()

person_timers = {}
person_id_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        logging.error("Cannot read frame. Check the connection.")
        break

    frame_counter += 1
    if frame_counter % process_every_nth_frame != 0:
        continue

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(frame_rgb)

    person_detections = []
    for detection in results[0].boxes.data:
        class_id = int(detection[-1])
        if class_id == 0:
            person_detections.append(detection)

    num_people = len(person_detections)

    if datetime.now() >= last_saved_time + timedelta(seconds=5):
        save_people_count(num_people)
        last_saved_time = datetime.now()

    annotated_frame = frame.copy()
    current_boxes = set()

    for detection in person_detections:
        x1, y1, x2, y2, conf, cls_id = detection[:6]
        box_id = (int(x1), int(y1), int(x2), int(y2))
        current_boxes.add(box_id)

        if box_id not in person_timers:
            person_timers[box_id] = (datetime.now(), person_id_counter)
            person_id_counter += 1

        entry_time, person_id = person_timers[box_id]
        duration = (datetime.now() - entry_time).total_seconds()

        cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(annotated_frame, f'Person: {conf:.2f} Time: {duration:.2f}s', 
                    (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    for box_id in list(person_timers.keys()):
        if box_id not in current_boxes:
            entry_time, person_id = person_timers.pop(box_id)
            duration = (datetime.now() - entry_time).total_seconds()
            save_person_timer(person_id, entry_time, duration)

    cv2.imshow("YOLO RTSP Stream - People Only", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
