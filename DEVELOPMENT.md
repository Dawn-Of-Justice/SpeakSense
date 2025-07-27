# SpeakSense Development Guide

## 📁 Project Structure Overview

The SpeakSense project has been reorganized into a clean, modular structure:

### Backend (`/backend/`)
- **Purpose**: Python-based AI/ML backend services
- **Tech Stack**: FastAPI, PyTorch, TensorFlow, Whisper, spaCy
- **Structure**:
  - `src/models/`: ML model implementations (ASD, audio classification)
  - `src/services/`: Core services (LLM integration, transcription)
  - `src/api/`: FastAPI endpoints and WebSocket handlers
  - `src/utils/`: Utility functions and helpers
  - `tests/`: Unit and integration tests

### Frontend (`/frontend/`)
- **Purpose**: React-based user interface
- **Tech Stack**: Next.js 15, React 19, Tailwind CSS, Socket.IO
- **Structure**: Standard Next.js app directory structure

### Models (`/models/`)
- **Purpose**: Trained model files and weights
- **Contents**: Model checkpoints, weights, and pre-trained models

### Data (`/data/`)
- **Purpose**: Data storage and management
- **Structure**:
  - `raw/`: Original audio/video data
  - `processed/`: Processed features and outputs
  - `assets/`: Project assets and demos

### Scripts (`/scripts/`)
- **Purpose**: Utility scripts for data processing and maintenance
- **Contents**: Data utilities, testing scripts, and maintenance tools

### Notebooks (`/notebooks/`)
- **Purpose**: Jupyter notebooks for experimentation and analysis
- **Contents**: Training notebooks, data exploration, and model testing

## 🚀 Getting Started

### Option 1: Manual Setup

1. **Set up the entire project**:
```bash
python manage.py --setup
```

2. **Start backend server**:
```bash
python manage.py --start-backend
```

3. **Start frontend server** (in another terminal):
```bash
python manage.py --start-frontend
```

### Option 2: Docker Setup

1. **Use Docker Compose**:
```bash
python manage.py --docker
```

This will start both services:
- Backend: http://localhost:8000
- Frontend: http://localhost:3000

## 🔧 Development Workflow

### Backend Development

1. Navigate to backend directory:
```bash
cd backend
```

2. Activate virtual environment:
```bash
# Windows
venv\Scripts\activate

# Unix/Linux/macOS
source venv/bin/activate
```

3. Install new dependencies:
```bash
pip install package-name
pip freeze > requirements.txt
```

4. Run tests:
```bash
python -m pytest tests/
```

### Frontend Development

1. Navigate to frontend directory:
```bash
cd frontend
```

2. Install new dependencies:
```bash
npm install package-name
```

3. Run in development mode:
```bash
npm run dev
```

4. Build for production:
```bash
npm run build
```

## 📊 Model Management

### Audio Classification Model
- **Location**: `backend/src/models/audio/`
- **Purpose**: Classify if speech is directed at the assistant
- **Training**: Use notebooks in `/notebooks/` directory

### Active Speaker Detection (ASD)
- **Location**: `backend/src/models/asd/`
- **Purpose**: Identify who is speaking in video
- **Weights**: Stored in `/models/weights/`

### LLM Integration
- **Location**: `backend/src/services/LLM.py`
- **Purpose**: Natural language processing and response generation
- **Supported**: Groq, Ollama, and other LLM providers

## 🔄 Data Pipeline

1. **Raw Data**: Audio/video files in `/data/raw/`
2. **Processing**: Use scripts in `/scripts/` for data preparation
3. **Feature Extraction**: Automated through model services
4. **Training**: Notebooks in `/notebooks/` for model training
5. **Inference**: Real-time processing through backend services

## 🧪 Testing

### Backend Tests
```bash
cd backend
python -m pytest tests/ -v
```

### Frontend Tests
```bash
cd frontend
npm test
```

### Integration Tests
```bash
python manage.py --clean  # Clean temporary files
python manage.py --setup  # Fresh setup
# Run both backend and frontend, then test API endpoints
```

## 🚢 Deployment

### Production Deployment

1. **Using Docker**:
```bash
docker-compose -f docker-compose.prod.yml up -d
```

2. **Manual Deployment**:
   - Backend: Deploy FastAPI with uvicorn and reverse proxy
   - Frontend: Build with `npm run build` and serve static files

### Environment Variables

**Backend** (`.env`):
```env
API_HOST=localhost
API_PORT=8000
GROQ_API_KEY=your_key_here
WHISPER_MODEL_SIZE=base
DEBUG=False
```

**Frontend** (`.env.local`):
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000/ws
```

## 🔍 Monitoring and Logs

- **Backend Logs**: `backend/logs/speaksense.log`
- **Process Outputs**: `data/processed/`
- **Error Tracking**: Built into FastAPI endpoints

## 🛠️ Maintenance

### Regular Maintenance
```bash
# Clean temporary files
python manage.py --clean

# Update dependencies
cd backend && pip install -r requirements.txt --upgrade
cd frontend && npm update
```

### Performance Optimization
- Monitor model inference times
- Optimize WebSocket connections
- Cache processed features when possible

## 📚 Additional Resources

- **Documentation**: `/docs/` directory
- **Research Papers**: `/docs/reference/`
- **API Documentation**: Available at http://localhost:8000/docs when backend is running
- **Model Training**: See notebooks in `/notebooks/`

## 🤝 Contributing

1. Follow the existing project structure
2. Add tests for new features
3. Update documentation
4. Use the provided scripts for maintenance
5. Follow Python PEP 8 and JavaScript/TypeScript best practices
