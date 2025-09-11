import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import torch
from ultralytics import YOLO
from typing import Dict, Any, List
import logging
import os
import sys
# Add parent directory to path to import base_agent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.base_agent import BaseAgent


class VisionAgent(BaseAgent):
    """
    Advanced Vision Agent that combines:
    1. Your existing emotion detection model
    2. Object detection using YOLO
    3. Scene description capabilities
    """
    
    def __init__(self, emotion_model_path="models/emotion_model.h5"):
        super().__init__("VisionAgent", [
            "emotion_detection", 
            "object_detection", 
            "scene_analysis", 
            "image_preprocessing"
        ])
        
        self.emotion_model_path = emotion_model_path
        self.emotion_model = None
        self.object_model = None
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        
        # Initialize models
        self._load_emotion_model()
        self._load_object_detection_model()
        
        self.logger.info("VisionAgent initialized with emotion and object detection")
    
    def _load_emotion_model(self):
        """Load your existing emotion detection model"""
        try:
            if os.path.exists(self.emotion_model_path):
                self.emotion_model = tf.keras.models.load_model(self.emotion_model_path)
                self.logger.info(f"Emotion model loaded from {self.emotion_model_path}")
                self.logger.debug(f"[DEBUG] Emotion model input shape: {self.emotion_model.input_shape}")
            else:
                self.logger.warning(f"Emotion model not found at {self.emotion_model_path}")
                self.emotion_model = None
        except Exception as e:
            self.logger.error(f"Error loading emotion model: {str(e)}")
            self.emotion_model = None
    
    def _load_object_detection_model(self):
        """Load YOLO model for object detection"""
        try:
            self.object_model = YOLO('yolov8n.pt')  # Auto-downloads if not cached
            self.logger.info("YOLO object detection model loaded")
        except Exception as e:
            self.logger.error(f"Error loading object detection model: {str(e)}")
            self.object_model = None
    
    def preprocess_for_emotion(self, image):
        """Preprocess image for emotion detection (handles grayscale conversion)"""
        try:
            # Convert PIL Image to numpy array if needed
            if isinstance(image, Image.Image):
                image_np = np.array(image)
                # Check if image is grayscale (PIL 'L' mode → shape (H, W))
                if len(image_np.shape) == 2:
                    self.logger.info("PIL image is grayscale; converting to RGB for emotion model")
                    image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)  # Convert to 3-channel RGB
                else:
                    # Convert PIL's RGB → OpenCV's BGR (if model expects BGR)
                    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            else:
                image_np = image  # Assume non-PIL image is already numpy array
                if len(image_np.shape) == 2:  # Grayscale (non-PIL)
                    image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
            
            # Detect faces using Haar Cascade (requires grayscale)
            gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY) if len(image_np.shape) == 3 else image_np
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml') 
            faces = face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.05,  # Sensitive to small faces
                minNeighbors=2,    # Fewer false negatives
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            self.logger.debug(f"[DEBUG] Faces detected by Haar Cascade: {len(faces)}")
            
            processed_faces = []
            for (x, y, w, h) in faces:
                # Validate face coordinates (avoid out-of-bounds errors)
                if y + h > gray.shape[0] or x + w > gray.shape[1]:
                    self.logger.warning(f"Skipping face (coordinates exceed image dimensions): (x={x}, y={y}, w={w}, h={h})")
                    continue
                
                # Extract and resize face region
                face_region = image_np[y:y+h, x:x+w]
                face_resized = cv2.resize(face_region, (48, 48))  # Match model input size
                
                # Normalize pixel values (0-255 → 0-1)
                face_normalized = face_resized.astype(np.float32) / 255.0
                face_final = face_normalized.reshape(1, 48, 48, 3)  # Add batch dimension
            
                processed_faces.append({
                    'processed_image': face_final,
                    'face_coordinates': (x, y, w, h),
                    'face_image': face_region
                })
            
            return processed_faces
        except Exception as e:
            self.logger.error(f"Error during emotion preprocessing: {str(e)}")
            return []
    
    def detect_emotions(self, image):
        """
        Detect emotions using your trained model
        Returns: Dict with emotions, faces detected, and message
        """
        if self.emotion_model is None:
            return {"error": "Emotion model not loaded", "emotions": []}
        
        try:
            processed_faces = self.preprocess_for_emotion(image)
            
            if not processed_faces:
                return {"emotions": [], "message": "No faces detected"}
            
            emotion_results = []
            for face_data in processed_faces:
                # Predict emotion probabilities
                prediction = self.emotion_model.predict(face_data['processed_image'], verbose=0)
                emotion_probs = prediction[0]
                
                # Create emotion dictionary
                emotion_dict = {emotion: float(prob) for prob, emotion in zip(emotion_probs, self.emotion_labels)}
                
                # Get dominant emotion and confidence
                dominant_emotion = self.emotion_labels[np.argmax(emotion_probs)]
                confidence = float(np.max(emotion_probs))
                
                emotion_results.append({
                    'dominant_emotion': dominant_emotion,
                    'confidence': confidence,
                    'all_emotions': emotion_dict,
                    'face_location': face_data['face_coordinates']
                })
            
            return {
                "emotions": emotion_results,
                "faces_detected": len(emotion_results),
                "message": f"Detected {len(emotion_results)} face(s) with emotions"
            }
        
        except Exception as e:
            self.logger.error(f"Error in emotion detection: {str(e)}")
            return {"error": str(e), "emotions": []}
    
    def detect_objects(self, image):
        """
        Detect objects using YOLO model
        Returns: Dict with detected objects, count, and message
        """
        if self.object_model is None:
            return {"error": "Object detection model not loaded", "objects": []}
        
        try:
            # Convert PIL Image to numpy array if needed
            if isinstance(image, Image.Image):
                image_np = np.array(image)
                
                # Handle alpha channel (RGBA → RGB)
                if len(image_np.shape) == 3 and image_np.shape[2] == 4:  # RGBA
                    self.logger.debug("Removing alpha channel from RGBA image")
                    image_np = image_np[:, :, :3]  # Keep R, G, B; drop A
                
                # Convert grayscale PIL image to RGB (if needed)
                if len(image_np.shape) == 2:  # Grayscale (PIL 'L' mode)
                    self.logger.debug("Converting grayscale PIL image to RGB for YOLO")
                    image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
            else:
                image_np = image  # Assume non-PIL image is already numpy array
            
            # Validate image is 3-channel (RGB)
            if len(image_np.shape) == 2:  # Grayscale (non-PIL)
                image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
            elif len(image_np.shape) == 3 and image_np.shape[2] == 1:  # Single-channel grayscale
                image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
            elif len(image_np.shape) == 3 and image_np.shape[2] == 4:  # RGBA (non-PIL)
                self.logger.debug("Removing alpha channel from non-PIL RGBA image")
                image_np = image_np[:, :, :3]
            
            # Run YOLO detection
            results = self.object_model(image_np, verbose=False)
            
            detected_objects = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for i in range(len(boxes)):
                        box = boxes.xyxy[i].cpu().numpy()
                        confidence = float(boxes.conf[i].cpu().numpy())
                        class_id = int(boxes.cls[i].cpu().numpy())
                        class_name = self.object_model.names[class_id]
                        
                        detected_objects.append({
                            'object': class_name,
                            'confidence': confidence,
                            'bbox': box.tolist(),
                            'class_id': class_id
                        })
            
            detected_objects.sort(key=lambda x: x['confidence'], reverse=True)
            
            return {
                "objects": detected_objects,
                "objects_detected": len(detected_objects),
                "message": f"Detected {len(detected_objects)} object(s)"
            }
        
        except Exception as e:
            self.logger.error(f"Error in object detection: {str(e)}")
            return {"error": str(e), "objects": []}
    
    def analyze_scene(self, image):
        """
        Complete scene analysis combining emotions and objects
        """
        try:
            # Get emotion analysis
            emotion_results = self.detect_emotions(image)
            
            # Get object detection
            object_results = self.detect_objects(image)
            
            # Create scene description
            scene_description = self._generate_scene_description(emotion_results, object_results)
            
            return {
                "scene_analysis": {
                    "emotions": emotion_results,
                    "objects": object_results,
                    "description": scene_description,
                    "analysis_timestamp": self._get_timestamp()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in scene analysis: {str(e)}")
            return {"error": str(e)}
    
    def _generate_scene_description(self, emotions, objects):
        """
        Generate natural language description of the scene
        """
        try:
            description_parts = []
            
            # Describe emotions
            if emotions.get("emotions"):
                emotion_count = len(emotions["emotions"])
                if emotion_count == 1:
                    dominant = emotions["emotions"][0]["dominant_emotion"]
                    confidence = emotions["emotions"][0]["confidence"]
                    description_parts.append(f"I see 1 person who appears {dominant} (confidence: {confidence:.2f})")
                else:
                    description_parts.append(f"I see {emotion_count} people with various emotions:")
                    for i, emotion_data in enumerate(emotions["emotions"][:3]):  # Limit to 3 for brevity
                        dominant = emotion_data["dominant_emotion"]
                        description_parts.append(f"  Person {i+1}: {dominant}")
            else:
                description_parts.append("No faces detected in the image")
            
            # Describe objects
            if objects.get("objects"):
                object_count = len(objects["objects"])
                unique_objects = list(set([obj["object"] for obj in objects["objects"]]))
                
                if object_count <= 3:
                    obj_list = [obj["object"] for obj in objects["objects"]]
                    description_parts.append(f"Objects present: {', '.join(obj_list)}")
                else:
                    description_parts.append(f"Multiple objects detected including: {', '.join(unique_objects[:5])}")
                    if len(unique_objects) > 5:
                        description_parts.append(f"...and {len(unique_objects)-5} more types")
            else:
                description_parts.append("No specific objects detected")
            
            # Combine into coherent description
            full_description = ". ".join(description_parts) + "."
            
            return {
                "natural_language": full_description,
                "summary": {
                    "people_count": len(emotions.get("emotions", [])),
                    "object_count": len(objects.get("objects", [])),
                    "dominant_emotion": emotions.get("emotions", [{}])[0].get("dominant_emotion", "none") if emotions.get("emotions") else "none",
                    "main_objects": list(set([obj["object"] for obj in objects.get("objects", [])[:5]]))
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error generating scene description: {str(e)}")
            return {"natural_language": "Unable to generate scene description", "error": str(e)}
    
    def _handle_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle incoming messages for vision analysis
        """
        try:
            message_type = message.get("type", "unknown")
            
            if message_type == "analyze_image":
                # Expect image data in message
                image_data = message.get("image_data")
                if image_data is None:
                    return {
                        "status": "error",
                        "error": "No image data provided",
                        "agent": self.name
                    }
                
                # Perform analysis
                analysis_result = self.analyze_scene(image_data)
                
                return {
                    "status": "success",
                    "analysis": analysis_result,
                    "agent": self.name,
                    "message_type": "vision_analysis"
                }
            
            elif message_type == "detect_emotions_only":
                image_data = message.get("image_data")
                if image_data is None:
                    return {"status": "error", "error": "No image data provided", "agent": self.name}
                
                emotion_result = self.detect_emotions(image_data)
                return {
                    "status": "success",
                    "emotions": emotion_result,
                    "agent": self.name,
                    "message_type": "emotion_analysis"
                }
            
            elif message_type == "detect_objects_only":
                image_data = message.get("image_data")
                if image_data is None:
                    return {"status": "error", "error": "No image data provided", "agent": self.name}
                
                object_result = self.detect_objects(image_data)
                return {
                    "status": "success",
                    "objects": object_result,
                    "agent": self.name,
                    "message_type": "object_analysis"
                }
            
            elif message_type == "status":
                return {
                    "status": "success",
                    "agent_status": {
                        "name": self.name,
                        "capabilities": self.capabilities,
                        "emotion_model_loaded": self.emotion_model is not None,
                        "object_model_loaded": self.object_model is not None,
                        "ready": (self.emotion_model is not None) and (self.object_model is not None)  # Both models should be ready
                    },
                    "agent": self.name
                }
            
            else:
                # Fall back to base agent behavior
                return super()._handle_message(message)
                
        except Exception as e:
            self.logger.error(f"Error handling message: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "agent": self.name
            }
    
    def _get_timestamp(self):
        """Get current timestamp for analysis"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# Test function
def test_vision_agent():
    """
    Test the vision agent with a sample image
    """
    print("🔍 Testing Vision Agent...")
    
    try:
        # Initialize agent
        agent = VisionAgent()
        
        # Test status
        status_message = {"type": "status"}
        status_response = agent.process_message(status_message)
        print(f"✅ Agent Status: {status_response}")
        
        # Test emotion detection with a dummy image (replace with real image path)
        test_image_path = "data/images/test_face.jpg"  # Update this path
        test_image = Image.open(test_image_path).convert('L')  # Load grayscale
        
        # Test emotion detection
        emotion_result = agent.detect_emotions(test_image)
        print(f"✅ Emotion Detection Result: {emotion_result}")
        
        # Test object detection with the same image
        object_result = agent.detect_objects(test_image)
        print(f"✅ Object Detection Result: {object_result}")
        
        # Test scene analysis
        scene_result = agent.analyze_scene(test_image)
        print(f"✅ Scene Analysis Result: {scene_result}")
        
        print("✅ Vision Agent tests passed!")
        
    except Exception as e:
        print(f"❌ Error testing Vision Agent: {str(e)}")


if __name__ == "__main__":
    test_vision_agent()