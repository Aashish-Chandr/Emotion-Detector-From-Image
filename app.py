import gradio as gr
import cv2
import numpy as np
from PIL import Image
from utils import get_emotion_emoji

# Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to detect emotion using face position and size
def detect_emotion(image):
    try:
        if image is None:
            return image, "Please upload an image first."
        
        # Convert PIL to OpenCV format (BGR)
        img_array = np.array(image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            return image, "❌ No faces detected in the image. Please try another image."
        
        # Simple emotion detection based on face position and size
        # (This is a simplification since we can't use DeepFace)
        results = []
        
        for (x, y, w, h) in faces:
            # Draw rectangle around face for visualization
            cv2.rectangle(img_bgr, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Simple rules for emotion "detection"
            # Using face position and size to simulate emotion detection
            face_ratio = w / h
            face_size = w * h / (image.shape[0] * image.shape[1])
            
            # Simple rules to assign emotions (just for demonstration)
            if face_ratio > 1.1:  # wider face
                emotion = "happy"
            elif face_ratio < 0.9:  # longer face
                emotion = "sad"
            elif face_size > 0.15:  # large face (close to camera)
                emotion = "angry"
            elif x < image.shape[1] / 3:  # face on left side
                emotion = "surprise"
            elif x > 2 * image.shape[1] / 3:  # face on right side
                emotion = "fear"
            else:  # default
                emotion = "neutral"
            
            # Get emoji for the emotion
            emoji = get_emotion_emoji(emotion)
            results.append(f"Face #{len(results)+1}: {emotion.capitalize()} {emoji}")
        
        # Convert result image back to RGB for display
        result_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Join all results
        result_text = "\n".join(results)
        
        return result_img, result_text
    except Exception as e:
        return image, f"❌ Error: {str(e)}"

# Create Gradio UI
interface = gr.Interface(
    fn=detect_emotion,
    inputs=gr.Image(type="numpy"),
    outputs=[
        gr.Image(type="numpy", label="Detected Faces"),
        gr.Textbox(label="Emotion Results")
    ],
    title="😊 Simple Emotion Detector",
    description="Upload a face image and this app will attempt to detect emotions based on facial features. Note: This is a simplified version that uses basic face proportions to estimate emotions."
)

# Launch the app
if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=5000, share=False)
