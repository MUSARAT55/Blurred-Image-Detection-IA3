
BLURVISION AI — ADVANCED IA-3 PROJECT
Student: MEHRUNBEE DASTGEER MULLA
Register No.: 8080120

TOPIC:
Blurred Image Detection using Deep Learning

FEATURES:
- Modern dark AI dashboard
- Sidebar navigation
- Dashboard metric cards
- Clear vs Blurred visualization
- Image upload and preview
- CNN-based detection support
- Confidence score
- Blur score
- Image resolution and file size
- Processing-time metric
- Analytics page with charts
- Scan history stored in SQLite
- Clear History option
- About / Project Flow page
- Settings page
- Responsive layout for laptop/mobile

PROJECT FLOW:
Input Image → Preprocessing → Deep Learning Model → Analysis/Prediction → Result → Display

IMPORTANT:
The project only detects Clear vs Blurred images. It does not perform image deblurring.

PYCHARM SETUP:
1. Extract ZIP.
2. Open the extracted folder in PyCharm.
3. Recommended Python: 3.11.
4. Open Terminal:
   pip install -r requirements.txt
5. Put training images into:
   dataset/train/clear
   dataset/train/blurred
   dataset/validation/clear
   dataset/validation/blurred
6. Train:
   python train_model.py
7. Start application:
   python app.py
8. Open:
   http://127.0.0.1:5000

WITHOUT A TRAINED MODEL:
The application still runs and uses an OpenCV blur-score fallback so the interface can be demonstrated. For the actual IA-3 deep-learning demonstration, train and save the CNN model first.

ACADEMIC NOTE:
The dashboard and analytics are presentation features. The core task remains blurred image detection using deep learning.
