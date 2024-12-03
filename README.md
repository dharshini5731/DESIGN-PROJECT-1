# DESIGN-PROJECT-1

STEP 1: INSTALL THE REQUIRED DEPENDENCIES 
Before running the project, ensure you have all the necessary libraries 
installed. 
Core Libraries 
• os (for directory and file management) 
• time (for time-based operations) 
• pickle (for saving/loading trained models) 
• sqlite3 (for database management) 
Computer Vision 
• opencv-python-headless (OpenCV for image processing, 
headless for GUI-less environments) 
• mtcnn (for face detection using Multi-task Cascaded 
Convolutional Networks) 
• Pillow (for handling and displaying images) 
Machine Learning and Neural Networks 
• tensorflow (for deep learning and neural network operations) 
• keras-facenet (pre-trained FaceNet model for generating face 
embeddings) 
• scikit-learn (for machine learning algorithms like SVM and 
data preprocessing) 
Data Handling and Visualization 
• numpy (for numerical operations and array handling) 
GUI Development 
• tkinter (for creating the GUI interface) 
• ttk (for advanced widgets in tkinter) 
Others 
• queue (for thread-safe operations in database handling) 
• threading (for multi-threading operations) 
STEP 2: PREPARE THE DATASET 
Create a directory for your dataset, with each subdirectory containing 
images of different classes (individuals). 
STEP 3: RUN THE FACE EMBEDDINGS TRAINER PROGRAM 
Execute FaceEmbeddingsTrainer.py to generate the embeddings and 
train the SVM model. 
This will: 
• Generate embeddings for each face. 
• Train an SVM model to recognize faces. 
• Save the embeddings and SVM model. 
STEP 4: RUN THE FACE RECOGNITION SYSTEM GUI 
Execute the Python script FaceRecognitionGUI.py 
Home Tab 
• View basic instructions and usage. 
Monitoring Tab 
• Click Start Monitoring to begin face recognition using the 
webcam. 
• The system will recognize known faces and log them into the 
database and detect unknown faces, save their images, and 
trigger a pop-up alert. 
• Click Stop Monitoring to end the process. 
Logs Tab 
• View logs of known and unknown faces. 
• Double-click an unknown face entry to view its image.
