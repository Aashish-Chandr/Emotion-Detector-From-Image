import gradio as gr
import cv2
import numpy as np
from PIL import Image
import os
import requests
import json
import datetime
import uuid
import tempfile
from deepface import DeepFace
from database import add_emotion_record, get_all_emotion_records, get_emotion_record, delete_emotion_record
from utils import convert_to_opencv_format, get_emotion_emoji

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

# Using get_emotion_emoji from utils.py

# Function to analyze faces in image
def analyze_image(image, selected_emotion=None, save_to_db=True):
    """
    Analyze faces in the input image and detect emotions using DeepFace.
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
        
        # Create a temporary file to save the image for DeepFace
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            temp_path = temp_file.name
            cv2.imwrite(temp_path, img_bgr)
        
        # Detect faces using DeepFace
        try:
            # First attempt with DeepFace
            analysis = DeepFace.analyze(img_path=temp_path, 
                                       actions=['emotion'],
                                       enforce_detection=False,
                                       detector_backend='opencv')
            
            if not analysis:  # If no faces detected by DeepFace
                raise Exception("No faces detected by DeepFace")
                
            # DeepFace may return a single dict or a list of dicts
            if isinstance(analysis, dict):
                analysis = [analysis]
                
        except Exception as df_error:
            # Fallback to traditional Haar cascade
            faces = face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(30, 30)
            )
            
            if len(faces) == 0:
                return image, f"No faces detected in the image. DeepFace error: {str(df_error)}"
                
            # Create placeholder for manual detection
            analysis = []
            for face_coords in faces:
                analysis.append({
                    "region": {"x": int(face_coords[0]), "y": int(face_coords[1]), 
                              "w": int(face_coords[2]), "h": int(face_coords[3])},
                    "emotion": {"neutral": 1.0},
                    "dominant_emotion": "neutral"
                })
        
        # Clean up temp file
        try:
            os.unlink(temp_path)
        except:
            pass
        
        # Create a copy of image for drawing
        result_img = img_bgr.copy()
        
        # Track emotions and confidence for database
        face_emotions = {}
        face_confidences = {}
        
        # Process each detected face
        output_text = []
        for i, face_data in enumerate(analysis):
            face_id = i + 1
            
            # Get face coordinates
            if "region" in face_data:
                # From DeepFace
                x = face_data["region"]["x"]
                y = face_data["region"]["y"]
                w = face_data["region"]["w"]
                h = face_data["region"]["h"]
            else:
                # Fallback
                continue
            
            # Extract face region
            face_color = result_img[y:y+h, x:x+w]
            face_gray = cv2.cvtColor(face_color, cv2.COLOR_BGR2GRAY) if len(face_color.shape) == 3 else face_color
            
            # Draw rectangle around face
            cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Add face number
            cv2.putText(result_img, f"#{face_id}", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Detect facial features (for visualization)
            eyes = eye_cascade.detectMultiScale(face_gray, 1.1, 5)
            smile = smile_cascade.detectMultiScale(face_gray, 1.8, 20, minSize=(25, 25))
            
            # Draw detected eyes
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(face_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)
            
            # Draw detected smile
            for (sx, sy, sw, sh) in smile:
                cv2.rectangle(face_color, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2)
            
            # Check if there's a manually selected emotion for this face
            if selected_emotion and selected_emotion.startswith(f"face_{face_id}_"):
                emotion = selected_emotion.split('_')[2]
                confidence = 1.0  # User-selected emotions have 100% confidence
            else:
                # Use DeepFace emotion result
                if "dominant_emotion" in face_data and "emotion" in face_data:
                    emotion = face_data["dominant_emotion"]
                    confidence = face_data["emotion"].get(emotion, 0.5) / 100.0
                    if confidence <= 0 or confidence > 1:
                        confidence = 0.5  # Normalize if out of range
                else:
                    # Fallback
                    emotion = "neutral"
                    confidence = 0.5
            
            # Store for database
            face_emotions[str(face_id)] = emotion
            face_confidences[str(face_id)] = confidence
            
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
        
        # Save to database if requested
        if save_to_db:
            # Generate a unique image name
            image_name = f"analysis_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            
            # Save record to database
            add_emotion_record(
                image_name=image_name,
                face_count=len(analysis),
                emotions_detected=json.dumps(face_emotions),
                confidence_levels=json.dumps(face_confidences)
            )
        
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
    Upload an image containing faces, and this app will detect emotions using DeepFace, a powerful facial analysis library.
    The system uses pre-trained deep learning models for accurate emotion classification.
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

# Function to display detection history
def show_detection_history():
    """Retrieve and display past emotion detection history from database"""
    try:
        # Get all records from database
        records = get_all_emotion_records()
        
        if not records:
            return "No detection history found in the database."
        
        # Format records for display
        history_text = []
        for record in records:
            # Parse JSON strings from database
            emotions = json.loads(record.emotions_detected)
            confidences = json.loads(record.confidence_levels)
            
            # Format timestamp
            timestamp = record.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            
            # Start entry with header
            entry = f"ID: {record.id} | Time: {timestamp} | Faces: {record.face_count}\n"
            
            # Add details for each face
            for face_id, emotion in emotions.items():
                confidence = confidences.get(face_id, 0) * 100
                emoji = get_emotion_emoji(emotion)
                entry += f"  - Face #{face_id}: {emotion.capitalize()} {emoji} ({confidence:.1f}%)\n"
            
            # Add notes if available
            if record.notes:
                entry += f"  Notes: {record.notes}\n"
                
            entry += "----------------------------------------\n"
            history_text.append(entry)
        
        return "\n".join(history_text)
    
    except Exception as e:
        return f"Error retrieving detection history: {str(e)}"

# Function to delete a record
def delete_record(record_id):
    """Delete a specific record from the database"""
    try:
        # Convert to integer
        record_id = int(record_id)
        
        # Delete the record
        success = delete_emotion_record(record_id)
        
        if success:
            return f"Record #{record_id} successfully deleted.", show_detection_history()
        else:
            return f"Record #{record_id} not found.", show_detection_history()
    
    except Exception as e:
        return f"Error deleting record: {str(e)}", show_detection_history()

# Launch the app
if __name__ == "__main__":
    with gr.Blocks(title="Emotion Detection System") as app:
        gr.Markdown("# 😊 Emotion Detection System")
        
        # Create tabs for different features
        with gr.Tabs():
            # Main detection tab
            with gr.TabItem("Emotion Detection"):
                demo.render()
            
            # History tab
            with gr.TabItem("Detection History"):
                gr.Markdown("## 📋 Emotion Detection History")
                gr.Markdown("View and manage your past emotion detection records.")
                
                # Display history
                history_text = gr.Textbox(show_label=False, lines=20)
                
                with gr.Row():
                    refresh_btn = gr.Button("Refresh History")
                    
                    with gr.Column():
                        delete_id = gr.Number(label="Record ID to Delete", precision=0)
                        delete_btn = gr.Button("Delete Record")
                
                # Set up event handlers for history tab
                refresh_btn.click(
                    fn=show_detection_history,
                    inputs=None,
                    outputs=history_text
                )
                
                delete_btn.click(
                    fn=delete_record,
                    inputs=delete_id,
                    outputs=[gr.Textbox(visible=False), history_text]
                )
                
                # Load history on tab open
                app.load(
                    fn=show_detection_history,
                    inputs=None,
                    outputs=history_text
                )
    
    # Launch the combined app
    app.launch(server_name="0.0.0.0", server_port=5000)