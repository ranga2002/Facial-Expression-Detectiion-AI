import numpy as np
import os
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# Load your custom trained model
MODEL_PATH = os.path.join('fer_model', 'emotion_model.h5')
try:
    model = load_model(MODEL_PATH)
    print("Custom model loaded.")
except Exception as e:
    print(f"Failed to load custom model: {e}")
    model = None

# Load ResNet50 (for fallback or ensemble logic)
try:
    resnet = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
    use_resnet = True
    print("ResNet50 loaded (ImageNet weights).")
except Exception as e:
    print(f"ResNet50 load failed: {e}")
    use_resnet = False

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load MTCNN for face detection
try:
    from mtcnn.mtcnn import MTCNN
    detector = MTCNN()
    use_mtcnn = True
    print("Using MTCNN for face detection.")
except ImportError:
    print("MTCNN not found. Falling back to Haar Cascade.")
    use_mtcnn = False

# Preprocess Face from Image
def preprocess_face(image_path):
    if not image_path or not isinstance(image_path, str):
        print("Invalid image path.")
        return None

    img_cv = cv2.imread(image_path)
    if img_cv is None:
        print(f"Failed to read image: {image_path}")
        return None

    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    face_img = None

    if use_mtcnn:
        rgb_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(rgb_img)
        if faces:
            x, y, w, h = faces[0]['box']
            x, y = max(0, x), max(0, y)
            face_img = gray[y:y+h, x:x+w]
    else:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        if len(faces) > 0:
            x, y, w, h = faces[0]
            face_img = gray[y:y+h, x:x+w]

    if face_img is None or face_img.size == 0:
        print(" No face detected.")
        return None

    if face_img.shape[0] < 48 or face_img.shape[1] < 48:
        print("Face too small to process.")
        return None

    resized_face = cv2.resize(face_img, (48, 48))
    cleaned_path = image_path.replace('.jpg', '_processed.jpg')
    cv2.imwrite(cleaned_path, resized_face)
    return cleaned_path

# Predict Emotion
def predict_emotion(image_path, show_top2=True, threshold=0.3):
    if model is None or not os.path.exists(image_path):
        return "Model unavailable or image not found."

    img = Image.open(image_path).convert('L').resize((48, 48))
    img_arr = np.array(img) / 255.0
    img_arr = img_arr.reshape(1, 48, 48, 1)

    prediction = model.predict(img_arr)[0]
    top_indices = prediction.argsort()[-2:][::-1]
    top_emotions = [(emotion_labels[i], prediction[i]) for i in top_indices]

    if show_top2:
        print("Top Predictions:")
        for label, conf in top_emotions:
            print(f"→ {label}: {conf*100:.2f}%")

    if top_emotions[0][1] < threshold:
        print(f"Low confidence: {top_emotions[0][0]} ({top_emotions[0][1]*100:.2f}%)")
        return "Uncertain . Try a clearer face image."

    return top_emotions[0][0]

# Optional: ResNet50 Feature Extractor (for future ensemble)
def extract_features_with_resnet(image_path):
    if not use_resnet:
        return None

    try:
        img = Image.open(image_path).convert('RGB').resize((224, 224))
        img_arr = img_to_array(img)
        img_arr = np.expand_dims(img_arr, axis=0)
        img_arr = preprocess_input(img_arr)
        features = resnet.predict(img_arr)
        return features
    except Exception as e:
        print(f"ResNet feature extraction failed: {e}")
        return None

# Emotion Suggestions
def get_suggestion(emotion):
    suggestions = {
        'Happy': "Keep smiling! Spread the joy.",
        'Sad': "Take a break and listen to your favorite music.",
        'Angry': "Try deep breathing. Stay calm.",
        'Fear': "Talk to someone you trust. You're not alone.",
        'Surprise': "Wow! What surprised you?",
        'Disgust': "Let's shift focus to something positive.",
        'Neutral': "Stay focused. You're doing great!"
    }
    return suggestions.get(emotion, "You're doing great!")
