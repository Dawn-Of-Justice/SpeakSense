import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences # type:ignore
import ollama


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

# Define a simple neural network model using PyTorch Lightning
class SimpleNN(pl.LightningModule):
    def __init__(self, input_size, hidden_size, output_size, lr=0.001):
        super(SimpleNN, self).__init__()
        self.lr = lr
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=output_size)
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        acc = self.accuracy(y_pred, y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        acc = self.accuracy(y_pred, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val acc", acc, prog_bar=True)
        
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer




class AddressClassifierPt:
    
    def __init__(self, model_path=r"audio_model\best_model.ckpt"):
        device = torch.device('cpu')
        # Load the model from checkpoint
        self.model = SimpleNN.load_from_checkpoint(model_path, input_size=768, hidden_size=512, output_size=2)
        self.model = self.model.to(device)

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
        embs = ollama.embeddings("nomic-embed-text:latest",
                            prompt=text)

        embeddings = torch.tensor(embs.embedding, dtype=torch.float32)
        with torch.no_grad():
            out = F.softmax(self.model(embeddings), -1)
            # print(out)
            
        predicted_class = torch.argmax(out, dim = -1).item()
        # print(predicted_class)
        prediction_prob = out[predicted_class]
        
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