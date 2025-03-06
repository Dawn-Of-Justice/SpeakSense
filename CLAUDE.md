# SpeakSense Working Guidelines

## Project Overview
SpeakSense is a multimodal deep learning project that detects when a user is speaking to a virtual assistant by analyzing both audio and video in real time.

## Environment Setup
- Install dependencies: `pip install -r requirements.txt`
- Additional module requirements: `pip install -r ASD_BASED_ARCH/requirements.txt`
- For transcription: `pip install -r Live_transcription/requirements.txt`

## Code Structure
- `ASD_BASED_ARCH/`: Active Speaker Detection model
- `Live_transcription/`: Speech transcription modules
- `audio_model/`: Audio processing models
- `video_model/`: Video processing models
- `tests/`: Testing modules

## Testing
- Run tests: `python -m unittest discover tests`
- Single test: `python -m unittest tests/test_name.py`

## Code Style
- Use descriptive variable names and PEP 8 style guidelines
- Organize imports: standard libraries first, then third-party, then local modules
- Implement error handling with try/except blocks for I/O operations
- Use docstrings for functions and classes
- Include type hints where possible

## Running
- For ASD inference: `python ASD_BASED_ARCH/Testcode.py`
- For real-time processing: `python ASD_BASED_ARCH/realtime.py`