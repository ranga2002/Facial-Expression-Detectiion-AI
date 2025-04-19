from django.shortcuts import render, redirect
from django.http import JsonResponse
from .forms import ImageUploadForm
from .utils import predict_emotion, get_suggestion, preprocess_face
from openai import OpenAI
from datetime import datetime
import os
from dotenv import load_dotenv
import base64
from PIL import Image
from io import BytesIO

load_dotenv()

client = OpenAI(
  api_key="Insert your OpenAI API key here",
)

def decode_base64_image(data_url):
    format, imgstr = data_url.split(';base64,')
    ext = format.split('/')[-1]
    image_data = base64.b64decode(imgstr)
    image = Image.open(BytesIO(image_data)).convert('RGB')
    filename = f"captured_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
    path = os.path.join("media", filename)
    image.save(path)
    return path

def home(request):
    return render(request, 'emotion/home.html', {
        'now': datetime.now(),
        'user_name': request.session.get('user_name', 'Friend'),
        'last_emotion': request.session.get('last_emotion')
    })

def about(request):
    return render(request, 'emotion/about.html')

def chat_reply(request):
    if request.method != 'POST':
        return JsonResponse({'reply': "Invalid request method."})

    user_message = request.POST.get('message', '').strip()
    if not user_message:
        return JsonResponse({'reply': "Please enter a valid message."})

    # Fetch chat history or start new
    chat_history = request.session.get('chat_messages', [])

    # Add user input
    chat_history.append({
        "role": "user",
        "content": user_message,
        "timestamp": datetime.now().strftime('%H:%M:%S')
    })

    # Add system prompt only once
    messages = [{"role": msg["role"], "content": msg["content"]} for msg in chat_history]
    if not any(msg["role"] == "system" for msg in messages):
        name = request.session.get("user_name", "Friend")
        emotion = request.session.get("last_emotion", "Neutral")
        messages.insert(0, {
            "role": "system",
            "content": f"You are a supportive AI assistant. The user is {name}, feeling {emotion}. Respond warmly and concisely."
        })

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        ai_reply = response.choices[0].message.content.strip()

        chat_history.append({
            "role": "assistant",
            "content": ai_reply,
            "timestamp": datetime.now().strftime('%H:%M:%S')
        })
        request.session['chat_messages'] = chat_history

        return JsonResponse({'reply': ai_reply})

    except Exception as e:
        return JsonResponse({'reply': f"Sorry, an error occurred: {str(e)}"})

def calculate_score(answers):
    score = 0
    for ans in answers:
        ans = ans.lower()
        if "never" in ans or "not at all" in ans:
            score += 5
        elif "sometimes" in ans or "a little":
            score += 3
        elif "often" in ans:
            score += 2
        elif "always" in ans or "very":
            score += 1
        else:
            score += 3
    return score

def generate_openai_response(messages):
    try:
        messages.insert(1, {
            "role": "system",
            "content": "You're an AI mental health buddy. Respond with warmth and empathy, briefly (2–4 lines)."
        })
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        return response.choices[0].message.content
    except Exception as e:
        return "Sorry, I couldn't process your request. Please try again later."

def index(request):
    prediction = suggestion = openai_response = None
    score = percent_score = 0
    processed_path = None
    path = None 
    chat_messages = request.session.get('chat_messages', [])

    if request.method == 'POST':
        if 'clear_chat' in request.POST:
            request.session['chat_messages'] = []
            return redirect('home')

        if 'chat_message' in request.POST:
            user_input = request.POST.get('user_message')
            chat_messages = request.session.get('chat_messages', [])
            if chat_messages and len(chat_messages) < 10:
                chat_messages.append({
                    "role": "user",
                    "content": user_input,
                    "timestamp": datetime.now().strftime('%H:%M:%S')
                })
                reply = generate_openai_response([{"role": m["role"], "content": m["content"]} for m in chat_messages])
                chat_messages.append({
                    "role": "assistant",
                    "content": reply,
                    "timestamp": datetime.now().strftime('%H:%M:%S')
                })
                request.session['chat_messages'] = chat_messages
            return redirect('analyze')

        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            name = form.cleaned_data['name']
            gender = form.cleaned_data['gender']
            age = form.cleaned_data['age']
            answers = [
                form.cleaned_data['question1'],
                form.cleaned_data['question2'],
                form.cleaned_data['question3'],
                form.cleaned_data['question4']
            ]
            comment = form.cleaned_data['comment']

            # Webcam or file image
            webcam_image = request.POST.get('captured_image')
            if webcam_image:
                path = decode_base64_image(webcam_image)
            else:
                image = form.cleaned_data['image']
                path = f'media/{image.name}'
                with open(path, 'wb+') as f:
                    for chunk in image.chunks():
                        f.write(chunk)

            processed_path = preprocess_face(path)

            if processed_path:
                prediction = predict_emotion(processed_path)
                suggestion = get_suggestion(prediction)
                score = calculate_score(answers)
                percent_score = int((score / 20) * 100)

                request.session['user_name'] = name
                request.session['last_emotion'] = prediction

                # OpenAI
                system_prompt = (
                    f"User info: {name}, {gender}, {age} years old. "
                    f"Emotion: {prediction}. Score: {score}/20. "
                    f"Answers: {answers}. Comment: {comment}."
                )
                chat_messages = [{
                    "role": "system",
                    "content": system_prompt,
                    "timestamp": datetime.now().strftime('%H:%M:%S')
                }]
                first_reply = generate_openai_response([{"role": "system", "content": system_prompt}])
                chat_messages.append({
                    "role": "assistant",
                    "content": first_reply,
                    "timestamp": datetime.now().strftime('%H:%M:%S')
                })
                openai_response = first_reply
                request.session['chat_messages'] = chat_messages

            else:
                prediction = "No face detected"
                suggestion = "Try another image with a clear, front-facing face."
                request.session['chat_messages'] = []

    else:
        # Clear session and cookies on page reload
        request.session.flush()
        form = ImageUploadForm()

    return render(request, 'emotion/result.html', {
        'form': form,
        'prediction': prediction,
        'suggestion': suggestion,
        'score': score,
        'percent_score': percent_score,
        'chat_messages': chat_messages,
        'openai_response': openai_response,
        'remaining_messages': 5 - (len(chat_messages) // 2),
        'image_path': path if processed_path else None
    })