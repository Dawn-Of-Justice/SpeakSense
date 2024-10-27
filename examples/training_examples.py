import yaml
from pathlib import Path
from speaksense.training.trainer import Trainer
from speaksense.data.datasets import SpeakSenseDataset
from speaksense.models.audio_model import AudioLanguageModel
from speaksense.models.video_model import VideoModel
from speaksense.models.fusion_model import MultimodalFusionModel

def main():
    # Load configuration
    config_path = Path("config/training.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Create dataset
    dataset = SpeakSenseDataset(
        config["datasets"]["avspeech"]["path"],
        config["datasets"]["havic"]["path"],
        config["datasets"]["gaze"]["path"]
    )
    
    # Create model components
    audio_model = AudioLanguageModel(config)
    video_model = VideoModel(config)
    fusion_model = MultimodalFusionModel(config)
    
    # Create trainer
    trainer = Trainer(
        config=config,
        models={
            "audio": audio_model,
            "video": video_model,
            "fusion": fusion_model
        },
        dataset=dataset
    )
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    main()