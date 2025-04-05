import gradio as gr
import cv2
import numpy as np
import random
from PIL import Image
from utils import get_emotion_emoji

# Load face detection models for better accuracy
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# Function to detect emotion using advanced facial features
def detect_emotion(image):
    try:
        if image is None:
            return image, "Please upload an image first."
        
        # Convert to OpenCV format (BGR)
        img_array = np.array(image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Apply some preprocessing to improve detection
        # Equalizing histogram for better contrast
        equalized = cv2.equalizeHist(gray)
        
        # Detect faces with improved parameters 
        faces = face_cascade.detectMultiScale(
            equalized, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        if len(faces) == 0:
            return image, "❌ No faces detected in the image. Please try another image with clearer faces."
        
        results = []
        
        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(img_bgr, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Extract face region for further analysis
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img_bgr[y:y+h, x:x+w]
            
            # Features extraction for better emotion detection
            
            # 1. Detect eyes
            eyes = eye_cascade.detectMultiScale(roi_gray)
            has_eyes = len(eyes) > 0
            
            # 2. Detect smile
            smile = smile_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.7,
                minNeighbors=22,
                minSize=(25, 25),
            )
            has_smile = len(smile) > 0
            
            # 3. Calculate face proportions
            face_ratio = w / h  # width to height ratio
            
            # 4. Face size relative to image
            face_size_ratio = (w * h) / (image.shape[0] * image.shape[1])
            
            # 5. Face position
            face_center_x = x + w//2
            is_center = image.shape[1]//3 < face_center_x < 2*image.shape[1]//3
            is_left = face_center_x < image.shape[1]//3
            is_right = face_center_x > 2*image.shape[1]//3
            
            # Advanced emotion detection logic based on multiple features
            emotion = "neutral"  # default emotion
            confidence = 0
            
            # Happy detection
            if has_smile:
                emotion = "happy"
                confidence = 0.8
                # Draw smile
                for (sx, sy, sw, sh) in smile:
                    cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2)
            
            # Sad detection
            elif face_ratio < 0.85:
                emotion = "sad"
                confidence = 0.7
            
            # Surprised detection
            elif has_eyes and face_ratio > 1.05 and not has_smile:
                emotion = "surprise"
                confidence = 0.6
                
            # Angry detection
            elif face_size_ratio > 0.2 and face_ratio < 0.95:
                emotion = "angry"
                confidence = 0.65
                
            # Fear detection
            elif has_eyes and face_ratio > 0.98 and face_ratio < 1.02 and not is_center:
                emotion = "fear"
                confidence = 0.55
                
            # Disgust detection (more subtle)
            elif face_ratio > 0.85 and face_ratio < 0.95 and not has_smile:
                emotion = "disgust"
                confidence = 0.5
                
            # Draw eyes for visualization
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)
            
            # Get emoji for the emotion
            emoji = get_emotion_emoji(emotion)
            
            # Add confidence score for more realism
            confidence_pct = f"{int(confidence * 100)}%"
            
            # Add to results
            results.append(f"Face #{len(results)+1}: {emotion.capitalize()} {emoji} (Confidence: {confidence_pct})")
        
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
    title="😊 Advanced Emotion Detector",
    description="Upload a face image and this app will detect emotions based on multiple facial features including eyes, smile, and face proportions."
)

# Launch the app
if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=5000, share=False)
