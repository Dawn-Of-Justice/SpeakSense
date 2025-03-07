# SpeakSense
SpeakSense is a multimodal deep learning project that detects when a user is speaking to a virtual assistant by analyzing both audio and video in real time

## Phase 1: Data Collection & Preparation

### Collect multimodal training data
- Record video, audio, and transcripts of people talking to and around the robot
- Include diverse scenarios (directly addressing robot, talking nearby but not to robot)
- Label data with "addressing robot" vs "not addressing robot" classifications

### Feature extraction pipeline
- Implement the active speaker detection model (Liao et al.)
- Set up basic visual feature extraction (gaze, orientation)
- Configure audio preprocessing pipeline
- Establish transcription service integration

## Phase 2: Initial Model Development

### Build baseline model
- Implement a simple Bidirectional LSTM architecture
- Create input pipelines for each modality
- Design feature fusion mechanism
- Develop training and evaluation scripts

### Basic training and validation
- Train on clear-cut examples first
- Implement cross-validation strategy
- Establish baseline metrics for accuracy, latency, and resource usage

## Phase 3: Model Enhancement

### Improve feature engineering
- Refine visual features (add sustained gaze detection, orientation angles)
- Enhance audio features (directivity, voice characteristics)
- Develop linguistic feature extraction (pronoun detection, imperative forms)

### Architectural improvements
- Add attention mechanisms
- Implement hierarchical structure for modality processing
- Optimize layer configurations

### Advanced training techniques
- Implement curriculum learning
- Add data augmentation for edge cases
- Fine-tune hyperparameters

## Phase 4: System Integration

### Develop real-time processing pipeline
- Create efficient preprocessing modules
- Implement sliding window for contextual memory
- Design adaptive thresholding system

### Optimize for low-end devices
- Quantize model weights
- Implement model pruning
- Profile and optimize critical paths

### Create staged activation system
- Develop always-on lightweight monitoring
- Build trigger mechanism for full model activation
- Implement power management strategies

## Phase 5: Testing & Refinement

### Controlled environment testing
- Measure accuracy metrics in controlled settings
- Benchmark latency and resource usage
- Identify common failure cases

### Real-world testing
- Deploy prototype in various environments
- Collect user feedback on naturalism and responsiveness
- Log false positives and false negatives

### Model refinement
- Retrain with additional edge cases
- Fine-tune confidence thresholds
- Optimize for specific deployment environments

## Phase 6: Deployment & Learning

### Full system deployment
- Integrate with robot's main systems
- Implement logging for continuous improvement
- Develop update mechanism

### Continuous learning
- Add capability to learn from successful interactions
- Implement personalization for specific users
- Create feedback mechanism for misinterpretations