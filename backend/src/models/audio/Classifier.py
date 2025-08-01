import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
import tensorflow as tf
import pickle
from transformers import pipeline
from tensorflow.keras.preprocessing.sequence import pad_sequences # type:ignore
import ollama


class AddressClassifier:
    
    def __init__(self, model_path=r'audio_model\robot_addressing_classifier_new.h5', tokenizer_path=r'audio_model\tokenizer.pickle'):
        
        self.model = tf.keras.models.load_model(r'audio_model\robot_addressing_classifier_new.h5')
        # Load the tokenizer
        with open(r'audio_model\tokenizer.pickle', 'rb') as handle:
            self.tokenizer = pickle.load(handle)

    def classify_text(self, text, max_sequence_length=100):
        """
        Classify a single text input to determine if it's addressing a robot.
        
        Args:
            text: Text string to classify
            max_sequence_length: Maximum length for padding (should match training)
            
        Returns:
            Dictionary with prediction results
        """
        # Convert to sequence
        sequences = self.tokenizer.texts_to_sequences([text])
        
        # Pad sequence
        padded_sequence = pad_sequences(
            sequences,
            maxlen=max_sequence_length,
            padding='post'
        )
        
        # Make prediction
        prediction_prob = self.model.predict(padded_sequence)[0][0]
        predicted_class = 1 if prediction_prob > 0.5 else 0
        
        # Return result
        is_addressing_robot = (predicted_class == 0)
        
        return {
            'text': text,
            'is_addressing_robot': is_addressing_robot,
            'confidence': float(max(prediction_prob, 1 - prediction_prob))
        }



class AddressClassifierPt:
    def __init__(self, model_path=r"audio_model\best_model.ckpt"):
        try:
            # Attempt to load the model from checkpoint
            from pathlib import Path
            import os
            
            # Convert backslashes to forward slashes for cross-platform compatibility
            
            # model_name = "audio_model/distilbert-speaksense/checkpoint-3306"
            model_name = r"C:\Users\Rohit Francis\Documents\GitHub\SpeakSense\audio_model\distilbert-speaksense\checkpoint-3306"
            # Check if the model exists locally
            model_dir = Path(model_name)
            print(f"Does audio model dir exists model_dir.exists():{model_dir.exists()}")
            if model_dir.exists():
                self.classifier = pipeline("text-classification", model=str(model_dir), tokenizer=str(model_dir))
                self.model_available = True
                print(f"✅ Address classifier loaded from {model_dir}")
            else:
                print(f"⚠️ Audio model not found at {model_dir}. Using fallback classifier.")
                assert "Audio model not found"
                self.classifier = None
                self.model_available = False
        except Exception as e:
            print(f"⚠️ Failed to load audio classifier: {e}")
            print("   Using fallback classifier that considers all input as addressing the robot.")
            self.classifier = None
            self.model_available = False

    def classify_text(self, text, max_sequence_length=100):
        """
        Classify a single text input to determine if it's addressing a robot.
        
        Args:
            text: Text string to classify
            max_sequence_length: Maximum length for padding (should match training)
            
        Returns:
            Dictionary with prediction results
        """
        if self.model_available and self.classifier:
            try:
                output = self.classifier(text)
                
                # Return result
                is_addressing_robot = (output[0]['label'] == "LABEL_0")
                
                return {
                    'text': text,
                    'is_addressing_robot': is_addressing_robot,
                    'confidence': output[0]['score']
                }
            except Exception as e:
                print(f"⚠️ Error during classification: {e}")
                # Fall through to fallback
        
        # Fallback: assume all text is addressing the robot with moderate confidence
        return {
            'text': text,
            'is_addressing_robot': True,
            'confidence': 0.7  # Moderate confidence for fallback
        }


# Example usage
if __name__ == "__main__":
    # Test with different examples
    test_examples = [
        "Hey robot, what's the weather today?",
        "I need to finish my homework soon.",
        "Robot, can you help me with this?",
        "The meeting starts at 2 PM.",
        "So, yeah this is actually a test for this way, but test for this weight burden detection. It is not a weight burden detection system."
    ]
    
    classifier = AddressClassifier()
    
    
    for text in test_examples:
        result = classifier.classify_text(text)
        status = "IS" if result['is_addressing_robot'] else "is NOT"
        print(f"Text: \"{text}\"")
        print(f"Result: This {status} addressing the robot")
        print(f"Confidence: {result['confidence']:.2f}")
        print()