import torch
from nemo.collections.asr.models import EncDecSpeakerLabelModel
from nemo.collections.asr.parts.utils.diarization_utils import OfflineDiarWithASR
import os

# 1. Diarization pipeline
def diarize_audio(audio_file):
    # Initialize the diarization pipeline
    diarization = OfflineDiarWithASR(
        asr_model_name="stt_en_conformer_ctc_large",  # ASR model for transcription
        speaker_model_name="speakerverification_speakernet",  # Speaker model for embeddings
        msdd_model_name="diar_msdd_telephonic",  # Diarization model
    )
    
    # Diarize the audio file
    diarization_manifest = diarization(audio_file)
    return diarization_manifest

# 2. Load the speaker model
def extract_embeddings(audio_segments):
    # Initialize SpeakerNet model
    model = EncDecSpeakerLabelModel.from_pretrained(model_name="speakerverification_speakernet")
    embeddings = []
    
    # Loop through each audio segment and get embeddings
    for segment in audio_segments:
        audio_tensor = torch.tensor(segment["audio_signal"]).unsqueeze(0)  # Add batch dimension
        length = torch.tensor([audio_tensor.shape[1]])
        embedding = model.get_embedding(audio_signal=audio_tensor, length=length)
        embeddings.append((segment["speaker_label"], embedding))
    return embeddings

# 3. Main processing
if __name__ == "__main__":
    audio_file = r"D:\main project\dataset_release_mm15\Friends.S01E03\speaking-audio\chandler.wav"
    
    # Step 1: Diarize audio
    diarized_segments = diarize_audio(audio_file)
    print("Diarized Segments:", diarized_segments)

    # Step 2: Extract embeddings for each speaker
    speaker_embeddings = extract_embeddings(diarized_segments)
    for speaker_label, embedding in speaker_embeddings:
        print(f"Speaker {speaker_label} - Embedding shape: {embedding.shape}")
