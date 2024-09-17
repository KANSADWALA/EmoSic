from flask import Flask, request, jsonify, render_template
import openai
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained ResNet model for emotion prediction
model = load_model("Resnet_model_version_2.keras")

# OpenAI GPT API Key (replace with your key)
openai.api_key = "YOUR_OPENAI_API_KEY"

# Define a function to preprocess the face image
def preprocess_image(image, target_size):
    image = image.resize(target_size)
    image = np.array(image)
    if image.shape[2] == 4:  # Remove alpha channel if present
        image = image[:, :, :3]
    image = np.expand_dims(image, axis=0)
    return image / 255.0  # Normalize the image

# Function to generate music recommendations using GPT-4
def get_music_recommendation(emotion):
    prompt = f"Suggest me a list of songs for someone feeling {emotion}. Provide 5 song recommendations."
    
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=150
    )
    
    # Extract the music recommendations from the response
    music_recommendations = response.choices[0].text.strip()
    
    return music_recommendations

# Define route for uploading the face image
@app.route("/", methods=["GET", "POST"])
def upload_image():
    if request.method == "POST":
        # Check if a file is uploaded
        if "file" not in request.files:
            return "No file uploaded", 400

        file = request.files["file"]

        # Open the image
        img = Image.open(file)

        # Preprocess the image for ResNet model
        processed_img = preprocess_image(img, target_size=(224, 224))  # Assuming ResNet expects 224x224 images

        # Make emotion prediction
        prediction = model.predict(processed_img)
        predicted_class = np.argmax(prediction, axis=-1)

        # Map predicted class to emotion labels
        emotion_labels = ["happy", "sad", "angry", "surprised", "neutral"]
        predicted_emotion = emotion_labels[predicted_class[0]]

        # Get music recommendations based on the predicted emotion using GPT
        music_recommendations = get_music_recommendation(predicted_emotion)

        # Render the result with the predicted emotion and music list
        return render_template("result.html", emotion=predicted_emotion, music=music_recommendations)
    
    # Render the upload page
    return render_template("upload.html")

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
