# 🎶EmoSic

## Overview
This Project involves emotion detection from face images and music recommendation tailored to the predicted emotion in a Flask app. The system uses a pre-trained ResNet model to analyze facial expressions and predict the corresponding emotion and integrates GPT-4 from OpenAI to generate a list of songs that match the user's emotional state.

## Features

<ul>
  <li> <strong>Emotion Detection:</strong> Leverages a ResNet model to detect emotions from uploaded face images.</li>
  
  <li> <strong>Music Recommendation:</strong> Provides personalized song recommendations based on the predicted emotion.</li>
  
  <li> <strong>User-Friendly Interface:</strong> The application has an intuitive, clean UI where users can easily upload images and receive music suggestions.</li>
</ul>

## Technologies Used
<ul>
  <li><strong>ResNet (v2):</strong> A deep residual network model used for detecting emotions from face images.</li>
  
  <li><strong>Flask:</strong> A lightweight web framework used for building the backend of the application, handling image uploads, and managing requests to predict emotions and recommend music.</li>

  <li> <strong>Keras & TensorFlow:</strong> Employed for deep learning tasks, including loading and using the ResNet model to process and analyze images.</li>

  <li> <strong>OpenCV:</strong> Utilized for image pre-processing, including face detection and alignment before feeding images to the ResNet model for emotion detection.</li>

  <li><strong>Pillow (PIL):</strong>  A Python library used to handle image manipulation, including converting user-uploaded images into formats suitable for deep learning models.</li>
  
  <li><strong>Bootstrap:</strong> Provides responsive and modern styling for the front-end, ensuring the web application has a user-friendly and visually appealing interface.</li>
  
  <li><strong>GPT-4 (LLM):</strong> A large language model used for generating personalized music recommendations based on the detected emotions, offering highly relevant suggestions tailored to the user’s emotional state.</li>
</ul>

## Dataset Used
Dataset: <a href="https://www.kaggle.com/datasets/msambare/fer2013">FER-2013</a>

## Project Structure

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>

<p>The directory structure of the project is as follows:</p>

<pre><code class="bash">
  EmoSic/
  │
  ├── kaggle.json                    # To the fetch the Dataset 
  │
  ├── EmoSic.ipynb                   # Notebook with ResNet model and emotion detection logic
  ├
  ├── Resnet_model_version_2.keras   # Pre-trained model for emotion detection
  │
  ├── app.py                         # Flask app to handle file uploads, predictions, and rendering templates
  │
  ├── templates
      ├── upload.html                # Image upload page
      └── result.html                # Result page with emotion and music recommendations
  
</code></pre>

</body>
</html>
  
## Contributing
Contributions are welcome! Please open an issue or submit a pull request.


## Contact
For any questions or suggestions, please open an issue or contact me at <a href="mailto:shubhamkansadwala@gmail.com">shubhamkansadwala@gmail.com</a>
.

<hr></hr>
