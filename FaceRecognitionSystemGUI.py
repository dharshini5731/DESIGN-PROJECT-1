import cv2 as cv
import numpy as np
import os
import sqlite3
import time
import pickle
from sklearn.preprocessing import LabelEncoder
from keras_facenet import FaceNet
import tkinter as tk
from tkinter import ttk, Toplevel
from PIL import Image, ImageTk
from datetime import datetime
import threading
import queue
import tkinter.messagebox as messagebox

# Constants
UNKNOWN_THRESHOLD = 1.0
DB_FILE = "face_logs.db"
HAARCASCADE_FILE = "haarcascade_frontalface_default.xml"

# Initialize FaceNet and Haar Cascade
facenet = FaceNet()
haarcascade = cv.CascadeClassifier(HAARCASCADE_FILE)

# Load SVM Model and Face Embeddings
faces_embeddings = np.load("faces_embeddings_done_2classes.npz")
X = faces_embeddings['arr_0']
Y = faces_embeddings['arr_1']
encoder = LabelEncoder()
encoder.fit(Y)
model = pickle.load(open("svm_model_160x160.pkl", 'rb'))

# Thread-safe queue for database operations
db_queue = queue.Queue()


def initialize_database():
    """Initialize the SQLite database with tables for known and unknown faces."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS known_faces (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT,
                        time TEXT,
                        date TEXT)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS unknown_faces (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        time TEXT,
                        date TEXT,
                        image_path TEXT)''')
    conn.commit()
    conn.close()


# Initialize the database
initialize_database()


class FaceRecognitionGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Face Recognition System")
        self.geometry("1000x650")
        self.configure(bg="#8174A0")
        self.stop_thread = False
        self.cap = None
        self.create_widgets()
        self.last_alert_time = 0  # Store the last alert time in seconds
        self.alert_interval = 120 

        # Start the database worker thread
        threading.Thread(target=self.db_worker, daemon=True).start()

    def create_widgets(self):
        # Title Label
        title_label = tk.Label(self, text="Face Recognition System", font=("Arial", 20, "bold"), bg="#8174A0", fg="white")
        title_label.pack(pady=10)

        # Notebook (Tabbed Layout)
        notebook = ttk.Notebook(self)
        notebook.pack(expand=True, fill="both")

        # Tabs
        self.home_frame = tk.Frame(notebook, bg="#EFB6C8")
        self.monitor_frame = tk.Frame(notebook, bg="#EFB6C8")
        self.logs_frame = tk.Frame(notebook, bg="#EFB6C8")

        notebook.add(self.home_frame, text="Home")
        notebook.add(self.monitor_frame, text="Monitoring")
        notebook.add(self.logs_frame, text="Logs")

        # Populate Pages
        self.create_home_page()
        self.create_monitoring_page()
        self.create_logs_page()

    def create_home_page(self):
        tk.Label(self.home_frame, text="Welcome to the Face Recognition System", font=("Arial", 18, "bold"), bg="#EFB6C8").pack(pady=30)
        tk.Label(self.home_frame, text="Use the Monitoring tab to start face recognition.", font=("Arial", 14), bg="#EFB6C8").pack(pady=10)
        tk.Label(self.home_frame, text="You can view logs of Known and Unknown faces in the Logs tab.", font=("Arial", 14), bg="#EFB6C8").pack(pady=10)

    def create_monitoring_page(self):
        tk.Button(self.monitor_frame, text="Start Monitoring", command=self.start_monitoring, font=("Arial", 12), bg="#FFD2A0").pack(pady=20)
        tk.Button(self.monitor_frame, text="Stop Monitoring", command=self.stop_monitoring, font=("Arial", 12), bg="#FFD2A0").pack(pady=20)

    def create_logs_page(self):
        tk.Label(self.logs_frame, text="Known Faces Logs", font=("Arial", 12, "bold"), bg="#EFB6C8").pack(pady=10)
        self.known_table = ttk.Treeview(self.logs_frame, columns=("Name", "Time", "Date"), show="headings", height=5)
        self.known_table.heading("Name", text="Name")
        self.known_table.heading("Time", text="Time")
        self.known_table.heading("Date", text="Date")
        self.known_table.pack(pady=10)

        tk.Label(self.logs_frame, text="Unknown Faces Log", font=("Arial", 12, "bold"), bg="#EFB6C8").pack(pady=10)
        self.unknown_table = ttk.Treeview(self.logs_frame, columns=("Time", "Date", "Image Path"), show="headings", height=5)
        self.unknown_table.heading("Time", text="Time")
        self.unknown_table.heading("Date", text="Date")
        self.unknown_table.heading("Image Path", text="Image Path")
        self.unknown_table.pack(pady=10)

        # Bind the double-click event to show_unknown_image
        self.unknown_table.bind("<Double-1>", self.show_unknown_image)

        self.refresh_logs()

    def start_monitoring(self):
        self.stop_thread = False
        self.cap = cv.VideoCapture(0)
        threading.Thread(target=self.recognize_faces).start()

    def stop_monitoring(self):
        self.stop_thread = True
        if self.cap:
            self.cap.release()
        cv.destroyAllWindows()

    def recognize_faces(self):
        while not self.stop_thread:
            ret, frame = self.cap.read()
            if not ret:
                continue

            gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            faces = haarcascade.detectMultiScale(gray_img, 1.3, 5)

            for x, y, w, h in faces:
                face_img = frame[y:y + h, x:x + w]
                face_img_resized = cv.resize(face_img, (160, 160))
                face_img_resized = np.expand_dims(face_img_resized, axis=0)
                ypred = facenet.embeddings(face_img_resized)
                face_name = model.predict(ypred)
                confidence = np.linalg.norm(ypred - X[face_name[0]])

                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                date_str, time_str = timestamp.split(' ')
                confidence_pct = 100 - int(confidence * 100)

                if confidence <= UNKNOWN_THRESHOLD:
                    final_name = encoder.inverse_transform(face_name)[0]
                    db_queue.put(("INSERT INTO known_faces (name, time, date) VALUES (?, ?, ?)", (final_name, time, date_str)))
                    self.draw_face_box(frame, (x, y, w, h), final_name, confidence_pct, color=(0, 255, 0))
                else:
                    # Get the current time in seconds
                    current_time = time.time()  
                
                # Check if enough time has passed since the last alert
                    if current_time - self.last_alert_time >= self.alert_interval:
                        file_path = f"unknown_faces/unknown_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                        cv.imwrite(file_path, frame[y:y + h, x:x + w])
                        db_queue.put(("INSERT INTO unknown_faces (time, date, image_path) VALUES (?, ?, ?)", (time_str, date_str, file_path)))
                        self.alert_unknown_face(file_path)
                        self.last_alert_time = current_time 
                    
                    self.draw_face_box(frame, (x, y, w, h), "Unknown", confidence_pct, color=(0, 0, 255))

            cv.imshow("Face Recognition", frame)
            if cv.waitKey(1) == 27:
                break
        self.stop_monitoring()

    def draw_face_box(self, frame, box, name, confidence_pct, color=(0, 255, 0)):
        x, y, w, h = box
        label = f"{name} ({confidence_pct}%)"
        cv.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv.putText(frame, label, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    def db_worker(self):
        while True:
            query, params = db_queue.get()
            conn = sqlite3.connect(DB_FILE)
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
            conn.close()
            db_queue.task_done()
    def alert_unknown_face(self, file_path):
        """Display a pop-up alert when an unknown person is detected."""
        top = Toplevel(self)
        top.title("Unknown Face Alert")
        top.geometry("300x300")
        top.configure(bg="#FF7043")
        tk.Label(top, text="Unknown Face Detected!", font=("Arial", 14, "bold"), bg="#FF7043").pack(pady=10)
        tk.Label(top, text=f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", font=("Arial", 12), bg="#FF7043").pack(pady=5)
        # Display unknown face image
        img = Image.open(file_path)
        img = img.resize((100, 100))
        img_tk = ImageTk.PhotoImage(img)
        img_label = tk.Label(top, image=img_tk)
        img_label.image = img_tk
        img_label.pack(pady=10)
        tk.Button(top, text="OK", command=top.destroy, bg="#FFAB91").pack(pady=10)
    

    def refresh_logs(self):
        self.known_table.delete(*self.known_table.get_children())
        self.unknown_table.delete(*self.unknown_table.get_children())

        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        cursor.execute("SELECT name, time, date FROM known_faces ORDER BY id DESC")
        for row in cursor.fetchall():
            self.known_table.insert("", "end", values=row)

        cursor.execute("SELECT time, date, image_path FROM unknown_faces ORDER BY id DESC")
        for row in cursor.fetchall():
            self.unknown_table.insert("", "end", values=row)

        conn.close()
        self.after(30000, self.refresh_logs)

    def show_unknown_image(self, event):
        """Display the image of the selected unknown face."""
        selected_item = self.unknown_table.selection()
        if selected_item:
            item = self.unknown_table.item(selected_item)
            image_path = item['values'][2]

            # Open the image in a new window
            img = Image.open(image_path)
            img = img.resize((500, 500))
            img_tk = ImageTk.PhotoImage(img)

            top = Toplevel(self)
            top.title("Unknown Face Image")
            img_label = tk.Label(top, image=img_tk)
            img_label.image = img_tk
            img_label.pack()


if __name__ == "__main__":
    app = FaceRecognitionGUI()
    app.mainloop()
