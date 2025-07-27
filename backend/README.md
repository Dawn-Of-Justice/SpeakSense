# Backend Configuration

## Environment Variables
Create a `.env` file in the backend directory with:

```
# API Configuration
API_HOST=localhost
API_PORT=8000
WEBSOCKET_PORT=8001

# Model Paths
AUDIO_MODEL_PATH=./src/models/audio/trained/
ASD_MODEL_PATH=./src/models/asd/weight/
WHISPER_MODEL_SIZE=base

# Groq API (if using)
GROQ_API_KEY=your_groq_api_key_here

# Audio Settings
SAMPLE_RATE=16000
CHUNK_SIZE=1024

# Logging
LOG_LEVEL=INFO
LOG_FILE=./logs/speaksense.log
```

## Running the Backend

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download spaCy model:
```bash
python -m spacy download en_core_web_sm
```

3. Run the main application:
```bash
python src/main.py
```

4. Or run the WebSocket server:
```bash
python src/api/fastapi_websocket_server.py
```

## Directory Structure

- `src/`: Main source code
  - `models/`: ML model implementations
  - `services/`: Core services (LLM, transcription)
  - `api/`: API endpoints and WebSocket handlers
  - `utils/`: Utility functions
- `tests/`: Unit and integration tests
- `config/`: Configuration files
- `logs/`: Application logs
