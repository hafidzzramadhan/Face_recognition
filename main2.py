import cv2
import face_recognition
import pandas as pd
import numpy as np
from datetime import datetime
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# Path untuk folder wajah
IMAGE_PATH = "images2"
attendance_file = "attendance.csv"

# Buffer untuk validasi multi-frame
recognition_buffer = {}
buffer_threshold = 5  # Jumlah frame yang diperlukan untuk konfirmasi

# Fungsi untuk memuat wajah yang dikenal
def load_known_faces():
    known_encodings = []
    known_names = []
    
    # Dictionary untuk dataset wajah
    people = {
        "Ranggi": ["Ranggi1.jpg", "Ranggi2.jpg","Ranggi3"],
        "Hj.Dr.Hafidz Ramadhan.S.kom": ["hafidz.jpg", "hafidz1.jpg"],
        "Ariel" : ["ariell.jpg"],
        "Michael": ["MIcahael Briant.jpg"],
        "Hasanul bashori": ["hasanul bashori.jpg"],
        "Mara sutan arrafi": ["Mara sutan.jpg"],
        "Jokowi dodo": ["jokowi.jpg"],
        "Partahi daniel":["Partahi Daniel.jpg"],
        "Mr.Agung sediyono":["Agung sediyono.jpg","Agung sediyono2","Agung sediyono3.jpg"],
        "Mrs.Binti solihah":["Binti solihah.jpg"],
        "Mr.Is mardianto":["Is_Mardianto.jpg"],
        
        
    }
    
    for name, images in people.items():
        encodings = []
        for img_name in images:
            img_path = f"{IMAGE_PATH}/{img_name}"
            try:
                img = face_recognition.load_image_file(img_path)
                encoding = face_recognition.face_encodings(img)
                if encoding:
                    encodings.append(encoding[0])
            except FileNotFoundError:
                print(f"File {img_path} tidak ditemukan.")
        
        # Rata-rata encoding untuk orang ini
        if encodings:
            avg_encoding = np.mean(encodings, axis=0)
            known_encodings.append(avg_encoding)
            known_names.append(name)
    return known_encodings, known_names

# Fungsi untuk mencatat kehadiran
def mark_attendance(name):
    try:
        df = pd.read_csv(attendance_file)
    except FileNotFoundError:
        df = pd.DataFrame(columns=["Name", "Date", "Time"])
    
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    
    if not ((df["Name"] == name) & (df["Date"] == date_str)).any():
        new_entry = pd.DataFrame([[name, date_str, time_str]], columns=["Name", "Date", "Time"])
        df = pd.concat([df, new_entry], ignore_index=True)
        df.to_csv(attendance_file, index=False)
        print(f"Kehadiran {name} tercatat!")
    else:
        print(f"{name} sudah tercatat hari ini.")

# Fungsi untuk validasi buffer
def validate_and_mark(name):
    if name not in recognition_buffer:
        recognition_buffer[name] = 1
    else:
        recognition_buffer[name] += 1
    
    if recognition_buffer[name] >= buffer_threshold:
        mark_attendance(name)
        del recognition_buffer[name]  # Reset buffer

# Load known faces
known_encodings, known_names = load_known_faces()

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break
    
    # Deteksi wajah menggunakan YOLOv8
    results = model(frame)
    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)  # Get bounding boxes
    
    for box in boxes:
        x1, y1, x2, y2 = box
        
        # Crop face dari frame
        face_frame = frame[y1:y2, x1:x2]
        
        # Konversi BGR ke RGB untuk face_recognition
        face_frame_rgb = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(face_frame_rgb)
        
        # Cocokkan wajah
        for encoding in encodings:
            matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=0.5)
            face_distances = face_recognition.face_distance(known_encodings, encoding)
            name = "Unknown"
            
            if matches:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_names[best_match_index]
                    confidence = (1 - face_distances[best_match_index]) * 100
                    validate_and_mark(name)
                    name += f" ({confidence:.2f}%)"
            
            # Gambar kotak dan nama
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Tampilkan frame secara real-time
    cv2.imshow("Face Recognition Attendance", frame)
    
    # Keluar dengan tombol 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
