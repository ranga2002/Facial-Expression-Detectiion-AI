
# Facial Expression Detection AI 😄🤖  
A Django-based AI web application to analyze human facial expressions using a webcam or uploaded image, and respond with personalized mental health suggestions powered by OpenAI.

---

## 👥 Group Members
- **Sri Ranga Bharadwaj Chakilam** – Machine Learning, FER2013, Model integration, OpenAI
- **Srilekha Kamineni** – UI/UX, Bootstrap, AOS, Animations, Camera Modal
- **Sai Kumar Gattu** – Backend support, Form logic, Deployment

---

## 📸 Features

- Upload an image or **capture a photo using webcam** via modal
- Emotion detection using a trained `.h5` model (CNN-based on FER2013)
- Real-time **OpenAI-powered chat** with mental health insights
- AOS scroll animations and floating chatbot
- Beautiful Bootstrap frontend with toggles and sliders
- **Personalized responses** based on name, age, gender, and form answers

---

## 🧠 Technologies Used

### 🔹 Backend
- Python
- Django
- TensorFlow (for model loading)
- OpenAI API (GPT-4o-mini)
- MTCNN / HaarCascade (Face detection)

### 🔸 Frontend
- HTML5, CSS3, JavaScript
- Bootstrap 5
- Font Awesome
- Animate.css, AOS.js

---

## 🗂️ Project Structure

```
Facial-Expression-Detectiion-AI/
│
├── emotion/             # Django app with forms, views, utils
├── fer_django_app/      # Django project root (settings, urls)
├── db.sqlite3           # Default database
├── manage.py
├── requirements.txt
```

---

## 📦 Installation & Setup

### 1. Clone this repo
```bash
git clone https://github.com/ranga2002/Facial-Expression-Detectiion-AI.git
cd Facial-Expression-Detectiion-AI
```

### 2. Create virtual environment *(optional but recommended)*

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up your OpenAI key

Create a `.env` file in the root with your API key:

```env
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

*(Note: This file is ignored by git.)*

---

## 🚀 Run the App

```bash
python manage.py runserver
```

Go to: `http://127.0.0.1:8000/`

---

## 📊 Dataset & Model

- Trained on: [FER2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013)
- Emotion categories: `['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']`
- Model: CNN trained in TensorFlow, saved as `emotion_model.h5`

---

## 🛠 Future Enhancements

- Improve model accuracy with MobileNet or ResNet-based ensemble
- Host live on Render / Vercel / Railway
- Add emotion-wise history for user progress
- Integrate multi-language chat support

---

## 🔐 Note on Security

✅ OpenAI keys are stored in `.env` and **excluded from commits**.  
❌ Avoid hardcoding sensitive data in views or settings.

---

## 🌐 Links

- 🧠 [FER2013 Dataset on Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)

---

## 📬 Contact

For feedback or collaboration:
- `chakilamsriranga@gmail.com`

---

_This project was built with passion, purpose, and teamwork._ ✨
