import gradio as gr
import cv2
import numpy as np
from PIL import Image
import os
import requests

# Load Haar cascades for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# Function to analyze emotion based on facial features
def analyze_emotion(face_img, has_smile, has_eyes, face_ratio, face_size_ratio):
    """
    Analyze emotions based on detected facial features.
    Returns emotion name and confidence score.
    """
    # Initialize emotion scores
    emotion_scores = {
        "happy": 0.0,
        "sad": 0.0, 
        "angry": 0.0,
        "fear": 0.0,
        "surprise": 0.0,
        "disgust": 0.0,
        "neutral": 0.2  # Slight bias for neutral as default
    }
    
    # Happy indicators
    if has_smile:
        emotion_scores["happy"] += 0.7
    
    # Sad indicators
    if not has_smile and face_ratio < 0.9:
        emotion_scores["sad"] += 0.5
    
    # Angry indicators
    if not has_smile and face_ratio < 0.95 and face_size_ratio > 0.15:
        emotion_scores["angry"] += 0.6
    
    # Surprise indicators
    if has_eyes and (face_ratio > 1.05 or face_size_ratio > 0.2) and not has_smile:
        emotion_scores["surprise"] += 0.6
    
    # Fear indicators
    if has_eyes and 0.95 < face_ratio < 1.05 and not has_smile:
        emotion_scores["fear"] += 0.4
    
    # Disgust indicators
    if 0.9 < face_ratio < 1.0 and not has_smile:
        emotion_scores["disgust"] += 0.3
    
    # Find dominant emotion and confidence
    dominant_emotion = max(emotion_scores, key=emotion_scores.get)
    confidence = emotion_scores[dominant_emotion]
    
    # Default to neutral if confidence is too low
    if confidence < 0.3:
        dominant_emotion = "neutral"
        confidence = 0.3
    
    return dominant_emotion, confidence

# Function to get emotion emoji
def get_emotion_emoji(emotion):
    """Return appropriate emoji for each emotion"""
    emojis = {
        'happy': '😊',
        'sad': '😢',
        'angry': '😠',
        'fear': '😨',
        'surprise': '😲',
        'disgust': '🤢',
        'neutral': '😐'
    }
    return emojis.get(emotion.lower(), '❓')

# Function to analyze faces in image
def analyze_image(image, selected_emotion=None):
    """
    Analyze faces in the input image and detect emotions.
    If selected_emotion is provided, override the detection for a specific face.
    """
    if image is None:
        return None, "Please upload an image."
    
    try:
        # Handle URL images
        if isinstance(image, str):
            response = requests.get(image, stream=True)
            image = np.array(Image.open(response.raw))
        
        # Convert to numpy array and BGR for OpenCV processing
        img_array = np.array(image)
        
        # Handle different image formats
        if len(img_array.shape) == 2:  # Grayscale
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
            gray = img_array
        elif img_array.shape[2] == 4:  # Has alpha channel
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        else:  # Regular RGB
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Equalize histogram for better contrast
        gray = cv2.equalizeHist(gray)
        
        # Detect faces using Haar cascade
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        
        if len(faces) == 0:
            return image, "No faces detected in the image."
        
        # Create a copy of image for drawing
        result_img = img_bgr.copy()
        
        # Process each detected face
        output_text = []
        for i, (x, y, w, h) in enumerate(faces):
            face_id = i + 1
            
            # Extract face region
            face_gray = gray[y:y+h, x:x+w]
            face_color = result_img[y:y+h, x:x+w]
            
            # Draw rectangle around face
            cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Add face number
            cv2.putText(result_img, f"#{face_id}", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Detect facial features
            eyes = eye_cascade.detectMultiScale(face_gray, 1.1, 5)
            smile = smile_cascade.detectMultiScale(face_gray, 1.8, 20, minSize=(25, 25))
            
            # Draw detected eyes
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(face_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)
            
            # Draw detected smile
            for (sx, sy, sw, sh) in smile:
                cv2.rectangle(face_color, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2)
            
            # Calculate metrics for emotion detection
            has_eyes = len(eyes) > 0
            has_smile = len(smile) > 0
            face_ratio = w / h  # width to height ratio
            face_size_ratio = (w * h) / (image.shape[0] * image.shape[1])
            
            # Check if there's a manually selected emotion for this face
            if selected_emotion and selected_emotion.startswith(f"face_{face_id}_"):
                emotion = selected_emotion.split('_')[2]
                confidence = 1.0  # User-selected emotions have 100% confidence
            else:
                # Analyze emotion based on detected features
                emotion, confidence = analyze_emotion(
                    face_gray, has_smile, has_eyes, face_ratio, face_size_ratio
                )
            
            # Get emoji and format confidence
            emoji = get_emotion_emoji(emotion)
            confidence_pct = f"{int(confidence * 100)}%"
            
            # Add to results
            output_text.append(f"Face #{face_id}: {emotion.capitalize()} {emoji} (Confidence: {confidence_pct})")
            
            # Add emotion label to image
            cv2.putText(result_img, f"{emotion}", (x, y+h+25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Convert back to RGB for display
        result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        
        return result_rgb, "\n".join(output_text)
        
    except Exception as e:
        return image, f"Error during analysis: {str(e)}"

# Function to handle manual emotion correction
def correct_emotion(face_num, emotion, last_image):
    """Apply user-selected emotion to a specific face and re-analyze the image"""
    if last_image is None:
        return None, "Please analyze an image first before correcting emotions."
        
    # Format the selected emotion string
    selected = f"face_{face_num}_{emotion}"
    
    # Re-analyze the image with the selected emotion
    return analyze_image(last_image, selected)

# Create Gradio interface
with gr.Blocks(title="Interactive Emotion Detection") as demo:
    gr.Markdown("# 😊 Interactive Emotion Detector")
    gr.Markdown("""
    Upload an image containing faces, and this app will detect emotions using computer vision techniques.
    If the automatic detection is incorrect, you can manually select the correct emotion for each face.
    """)
    
    # Keep track of the last analyzed image
    last_analyzed_image = gr.State(value=None)
    
    with gr.Row():
        with gr.Column(scale=1):
            # Input controls
            input_image = gr.Image(label="Upload Image")
            analyze_btn = gr.Button("Analyze Image", variant="primary")
            
            # Manual correction controls
            with gr.Group():
                gr.Markdown("### Correct Detection")
                with gr.Row():
                    face_num = gr.Dropdown(
                        [1, 2, 3, 4, 5], 
                        label="Face Number", 
                        value=1
                    )
                    emotion = gr.Dropdown(
                        ["happy", "sad", "angry", "fear", "surprise", "disgust", "neutral"], 
                        label="Correct Emotion", 
                        value="happy"
                    )
                correct_btn = gr.Button("Apply Correction")
        
        with gr.Column(scale=1):
            # Output display
            output_image = gr.Image(label="Results")
            output_text = gr.Textbox(label="Detected Emotions")
    
    # Set up event handlers
    analyze_btn.click(
        fn=analyze_image,
        inputs=input_image,
        outputs=[output_image, output_text],
        preprocess=True
    ).then(
        # Store the analyzed image for later use with corrections
        fn=lambda img: img,
        inputs=input_image,
        outputs=last_analyzed_image
    )
    
    correct_btn.click(
        fn=correct_emotion,
        inputs=[face_num, emotion, last_analyzed_image],
        outputs=[output_image, output_text]
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=5000)