import cv2
import numpy as np

def convert_to_opencv_format(pil_image):
    """
    Convert a PIL Image to OpenCV format (BGR)
    
    Args:
        pil_image: PIL Image object
        
    Returns:
        OpenCV compatible image (BGR format)
    """
    # Convert PIL Image to numpy array (RGB)
    img_array = np.array(pil_image)
    
    # Convert RGB to BGR (OpenCV format)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    return img_bgr

def get_emotion_emoji(emotion):
    """
    Return an appropriate emoji for each emotion
    
    Args:
        emotion: Detected emotion as string
        
    Returns:
        Emoji character representing the emotion
    """
    emotion_emojis = {
        'happy': '😊',
        'sad': '😢',
        'angry': '😠',
        'fear': '😨',
        'surprise': '😲',
        'disgust': '🤢',
        'neutral': '😐'
    }
    
    return emotion_emojis.get(emotion.lower(), '❓')
