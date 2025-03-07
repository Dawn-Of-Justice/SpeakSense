import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences # type:ignore


class AddressClassifier:
    
    def __init__(self, model_path=r'audio_model\robot_addressing_classifier.h5', tokenizer_path=r'audio_model\tokenizer.pickle'):
        
        self.model = tf.keras.models.load_model(r'audio_model\robot_addressing_classifier.h5')
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

# Example usage
if __name__ == "__main__":
    # Test with different examples
    test_examples = [
        "Hey robot, what's the weather today?",
        "I need to finish my homework soon.",
        "Robot, can you help me with this?",
        "The meeting starts at 2 PM."
    ]
    
    classifier = AddressClassifier()
    
    
    for text in test_examples:
        result = classifier.classify_text(text)
        status = "IS" if result['is_addressing_robot'] else "is NOT"
        print(f"Text: \"{text}\"")
        print(f"Result: This {status} addressing the robot")
        print(f"Confidence: {result['confidence']:.2f}")
        print()