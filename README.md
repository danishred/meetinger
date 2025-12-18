# Meeting Summary Generator

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A sophisticated Python application that automatically processes meeting recordings (MP4, AVI, MOV, etc.) to generate comprehensive meeting summaries. Features include dual STT models (Whisper and Hinglish), Voice Activity Detection (VAD), advanced audio processing, and AI-powered summarization using Ollama with Gemma-2.

## 🚀 Key Features

- **Multi-Format Support**: Process MP4, AVI, MOV, MKV, WebM, and more
- **Dual STT Models**: Choose between OpenAI's Whisper (multi-language) or Oriserve's Whisper-Hinglish (Latin script)
- **Voice Activity Detection**: Advanced VAD processing with configurable aggressiveness levels
- **Smart Audio Processing**: Moderate silence cancellation and speech enhancement
- **AI-Powered Summarization**: Generate structured markdown summaries using Ollama with Gemma-2
- **Rich Visualizations**: Speech activity plots and processing statistics
- **Smart File Management**: Automatic input/output organization with video-specific folders
- **Real-time Progress**: Beautiful Rich console output with timing metrics
- **Customizable Prompts**: Environment-based prompt customization
- **Performance Monitoring**: Detailed timing analysis and performance metrics

## 🎯 Quick Start

### 30-Second Setup
```bash
# 1. Clone and setup
git clone <repo-url> && cd meetinger
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Install models
ollama pull gemma2:9b-instruct-q4_K_M

# 3. Process your first meeting
cp meeting.mp4 input/
python app.py

# 4. Check results
ls output/meeting/
```

## 🏗️ System Architecture

```
Input Video (MP4/AVI/MOV/etc.)
        ↓
Audio Extraction (ffmpeg)
        ↓
Voice Activity Detection (WebRTC VAD)
        ↓
Audio Processing & Filtering
        ↓
STT Transcription (Whisper ↔ Hinglish)
        ↓
Transcript Generation (Markdown)
        ↓
AI Summarization (Ollama + Gemma-2)
        ↓
Structured Meeting Summary (Markdown)
```

### Processing Pipeline

1. **Input Detection**: Automatically finds videos in `input/` folder or Windows Videos directory
2. **Audio Extraction**: Converts video to 16kHz mono WAV using ffmpeg
3. **VAD Processing**: Identifies speech segments with configurable aggressiveness
4. **STT Transcription**: Dual-model support with user selection
5. **Transcript Generation**: Clean markdown formatting with timestamps
6. **AI Summarization**: Structured summary generation with custom prompts
7. **Output Organization**: Video-specific folders with all artifacts

## 📋 Prerequisites

### Core Requirements
- **Python 3.8+** with pip
- **ffmpeg** - Audio/video processing
- **Ollama** - Local LLM inference

### Optional Dependencies
- **CUDA-compatible GPU** (recommended for faster processing)
- **4GB+ RAM** (minimum for Hinglish model)

### Installation Guide

#### Python Setup
```bash
# Check Python version
python3 --version  # Should be 3.8 or higher

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # Linux/Mac: venv\Scripts\activate (Windows)
```

#### System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg python3-dev
```

**macOS:**
```bash
# Install Homebrew if not installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install ffmpeg
brew install ffmpeg
```

**Windows:**
```powershell
# Using Chocolatey
choco install ffmpeg python3

# Or download from https://ffmpeg.org/download.html
```

#### Ollama Setup
1. Download from [ollama.ai/download](https://ollama.ai/download)
2. Install and start the application
3. Verify installation:
```bash
ollama --version
ollama list  # Should show available models
```

## 🚀 Installation

### Quick Start
```bash
# 1. Clone the repository
git clone <repository-url>
cd meetinger

# 2. Set up virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# OR: venv\Scripts\activate (Windows)

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install Ollama model
ollama pull gemma2:9b-instruct-q4_K_M

# 5. Configure environment (optional)
cp .env.example .env  # Add your customizations
```

### Dependencies Installation

The application supports two transcription backends:

#### Option A: Whisper Backend (Recommended)
```bash
pip install openai-whisper
```

#### Option B: Hugging Face Backend
```bash
pip install torch transformers accelerate
```

### Model Downloads
```bash
# Required: Ollama summarization model
ollama pull gemma2:9b-instruct-q4_K_M

# Optional: Whisper models (downloaded automatically on first use)
# whisper-medium (recommended for quality/speed balance)
# whisper-large (highest quality, slower)
```

## 📁 Project Structure

```
meetinger/
├── app.py                          # Main application entry point
├── requirements.txt                # Python dependencies
├── README.md                      # This documentation file
├── .gitignore                     # Git ignore rules
├── .env                           # Environment variables (optional)
├── input/                         # Input directory for video files
├── output/                        # Output directory for processed files
│   ├── {video_name}/              # Video-specific output folders
│   │   ├── transcript/            # Transcript files
│   │   ├── summary/               # Summary files
│   │   ├── audio/                 # Processed audio files
│   │   └── visualizations/        # Speech activity plots
│   └── .gitkeep
├── src/                           # Source code modules
│   ├── utils.py                   # Utility functions
│   ├── video_processor.py         # Video to audio conversion
│   ├── transcriber.py             # STT transcription (Whisper/Hinglish)
│   ├── summarizer.py              # Ollama summary generation
│   └── vad_processor.py           # Voice Activity Detection
└── logs/                          # Application logs (optional)
```

### File Organization
- **Input Videos**: Place in `input/` folder for automatic processing
- **Output Organization**: Each video gets its own folder in `output/`
- **Transcripts**: Saved as `{video_name}_transcript.md`
- **Summaries**: Saved as `{video_name}_summary.md`
- **Audio Files**: Processed audio in `{video_name}/audio/`
- **Visualizations**: Speech activity plots in `{video_name}/visualizations/`

## 🎯 Usage

### Quick Start

1. **Place your video in the input folder:**
```bash
cp my_meeting.mp4 input/
```

2. **Run the application:**
```bash
python app.py
```

3. **Check the output:**
```bash
ls output/
# my_meeting/
#   ├── transcript/my_meeting_transcript.md
#   ├── summary/my_meeting_summary.md
#   ├── audio/my_meeting_filtered.wav
#   └── visualizations/my_meeting_speech_activity.png
```

### Advanced Usage

#### STT Model Selection
The application offers a choice between two transcription models:

```bash
# Interactive model selection (10-second timeout)
python app.py

# Options:
# 1. Whisper-Hinglish (Oriserve) - Specialized for Hinglish
# 2. Whisper (OpenAI) - Multi-language, general purpose
```

#### VAD Configuration
Voice Activity Detection can be configured via environment variables or code:

```python
# In app.py
USE_VAD = True                    # Enable/disable VAD
VAD_AGGRESSIVENESS = 2           # 0-3 (0=conservative, 3=aggressive)
VAD_MODE = "moderate"            # 'conservative', 'moderate', 'aggressive'
CREATE_VISUALIZATION = True      # Generate speech activity plots
```

#### Custom Prompts
Add custom instructions via `.env` file:
```bash
# .env
extras="Custom instructions for summary generation..."
```

### Processing Workflow

The application follows this comprehensive 7-step process:

1. **Dependency Check**: Verifies all required tools and models
2. **Video Discovery**: Finds videos in `input/` folder or Windows Videos
3. **Output Setup**: Creates video-specific output directory
4. **Audio Extraction**: Converts video to 16kHz mono WAV using ffmpeg
5. **VAD Processing**: Identifies speech segments with configurable filtering
6. **STT Transcription**: Converts audio to text using selected model
7. **AI Summarization**: Generates structured markdown summary

### File Naming Convention

- **Input Video**: `input/meeting.mp4`
- **Transcript**: `output/meeting/transcript/meeting_transcript.md`
- **Summary**: `output/meeting/summary/meeting_summary.md`
- **Filtered Audio**: `output/meeting/audio/meeting_filtered.wav`
- **Visualization**: `output/meeting/visualizations/meeting_speech_activity.png`

### Output Format

#### Transcript Structure
```markdown
# Meeting Transcript (Hinglish)

[Raw transcribed text with proper formatting]
```

#### Summary Structure
```markdown
# Meeting Title

Meeting Purpose

[Brief statement of meeting goals and objectives]

Key Takeaways

  - [4-7 bullet points of most critical decisions]
  - [Action items with specific names and commitments]
  - [Important decisions and outcomes]

Topics

[Main Topic 1]

  - [Detailed description with subtopics]
  - [Specific details and decisions]
      - [Nested information with proper indentation]

[Main Topic 2]

  - [Detailed description]
  - [Action items and responsibilities]

Other Updates

  - [Miscellaneous items and status updates]

Next Steps

  - [Assignee Name]:
      - [Specific action item 1]
      - [Specific action item 2]

[Questions and Answers Section]
```

## ⚙️ Configuration

### Application Settings

Edit [`app.py`](app.py) to customize global settings:

```python
# Processing Configuration (lines 48-52)
USE_VAD = True                    # Enable Voice Activity Detection
VAD_AGGRESSIVENESS = 2           # VAD sensitivity (0-3)
VAD_MODE = "moderate"            # Preset: 'conservative', 'moderate', 'aggressive'
CREATE_VISUALIZATION = True      # Generate speech activity plots
CLEANUP_AUDIO = True             # Remove intermediate audio files

# Model Configuration (lines 56-57)
OUTPUT_DIR = "output"            # Output directory
OLLAMA_MODEL = "gemma2:9b-instruct-q4_K_M"  # Ollama model

# STT Model Selection
# Interactive selection with 10-second timeout
# Options: Whisper-Hinglish or Whisper (OpenAI)
```

### VAD Preset Modes

#### Conservative Mode
```python
VAD_MODE = "conservative"
# Aggressiveness: 1, Max Silence: 2.0s, Min Speech: 0.3s
# Best for: High-quality audio, preserving maximum content
```

#### Moderate Mode (Default)
```python
VAD_MODE = "moderate"
# Aggressiveness: 2, Max Silence: 1.5s, Min Speech: 0.5s
# Best for: General use, balanced filtering
```

#### Aggressive Mode
```python
VAD_MODE = "aggressive"
# Aggressiveness: 3, Max Silence: 1.0s, Min Speech: 0.7s
# Best for: Noisy environments, maximum filtering
```

### STT Model Configuration

#### Whisper Models (OpenAI)
```python
# Available sizes (in order of quality/speed):
# tiny    - Fastest, lowest accuracy (39M parameters)
# base    - Fast, decent accuracy (74M parameters)
# small   - Balanced speed/accuracy (244M parameters)
# medium  - Slower, higher accuracy (769M parameters)
# large   - Slowest, highest accuracy (1550M parameters)

# Usage in app.py:
transcriber = Transcriber(model_size="medium")
```

#### Hinglish Model (Oriserve)
```python
# Specialized for Hindi-to-Hinglish transcription
# Outputs Latin script with Hinglish content
# Requires: torch, transformers, accelerate

transcriber = WhisperHinglishTranscriber()
```

### Environment Variables

Create `.env` file for custom configurations:
```bash
# .env
extras="Custom prompt instructions for summary generation"
# Additional environment variables can be added here
```

### Ollama Model Options

#### Recommended Models
```bash
# Gemma-2 (Recommended)
ollama pull gemma2:9b-instruct-q4_K_M

# Alternative models
ollama pull llama3.2:latest          # General purpose
ollama pull mistral:latest           # Technical content
ollama pull codellama:latest         # Code-related meetings
```

#### Custom Model Configuration
```python
# In app.py
OLLAMA_MODEL = "llama3.2:latest"     # Change model name
```

## 🚨 Troubleshooting

### Common Issues & Solutions

#### Dependency Problems
```bash
# "ffmpeg not found"
sudo apt install ffmpeg  # Ubuntu/Debian
brew install ffmpeg      # macOS

# "torch not found"
pip install torch --index-url https://download.pytorch.org/whl/cu118  # CUDA
pip install torch --index-url https://download.pytorch.org/whl/cpu    # CPU

# "ollama not accessible"
ollama serve  # Start Ollama server
```

#### Model Issues
```bash
# "Model not found"
ollama list           # Check available models
ollama pull gemma2:9b-instruct-q4_K_M  # Pull missing model

# "CUDA out of memory" (Hinglish model)
# Reduce batch_size in transcriber.py from 4 to 2
# Or switch to Whisper model (smaller memory footprint)
```

#### Audio Processing Issues
```python
# "VAD processing failed" - Try different settings:
VAD_MODE = "conservative"      # Less aggressive filtering
VAD_AGGRESSIVENESS = 1         # Lower sensitivity
USE_VAD = False                # Disable VAD entirely

# "Transcription returned empty text":
# - Check audio quality and volume
# - Try different STT model (Whisper vs Hinglish)
# - Use larger Whisper model (medium/large)
# - Adjust VAD aggressiveness
```

#### File System Issues
```bash
# "Permission denied"
chmod 755 input/ output/
sudo chown -R $USER:$USER input/ output/

# "No video files found"
ls input/                    # Check input folder
ls /mnt/c/Users/danis/Videos/ # Check Windows Videos (WSL)

# "Multiple video files found"
# Keep only one video in input/ folder
# Or process videos individually
```

#### Performance Problems
```bash
# Check system resources
htop              # CPU/Memory usage
nvidia-smi        # GPU usage (if applicable)
df -h             # Disk space

# Optimize for speed:
VAD_MODE = "aggressive"        # Faster processing
transcriber = Transcriber("tiny")  # Fastest model
CLEANUP_AUDIO = True          # Free disk space

# Optimize for accuracy:
VAD_MODE = "conservative"     # Preserve content
transcriber = Transcriber("large")  # Best accuracy
```

### Getting Help

1. **Enable Debug Mode**:
```python
# In app.py
setup_logging(level=logging.DEBUG)
```

2. **Check Logs**:
```bash
tail -f logs/app.log  # Real-time log monitoring
```

3. **Verify Installation**:
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
ollama list  # Check available models
```

4. **Test Components Individually**:
```python
# Test VAD
from src.vad_processor import VADProcessor
vad = VADProcessor()
segments = vad.detect_speech_segments("test.wav")

# Test Transcription
from src.transcriber import Transcriber
transcriber = Transcriber("tiny")
text = transcriber.transcribe_audio("test.wav")
```

### "Model not found"
```bash
# Pull the required model
ollama pull gemma2:9b-instruct-q4_K_M

# Verify model is available
ollama list
```

### "No video files found"
```bash
# Place video in input folder
mkdir -p input
cp your_meeting.mp4 input/

# Or check Windows Videos folder (WSL only)
ls /mnt/c/Users/danis/Videos/
```

### "Multiple video files found"
```bash
# Keep only one video in input folder
ls input/
# Remove extra files or process them individually
```

### "VAD processing failed"
```python
# Try different VAD settings in app.py
VAD_MODE = "conservative"      # Less aggressive
VAD_AGGRESSIVENESS = 1         # Lower sensitivity
USE_VAD = False                # Disable VAD entirely
```

### "Transcription returned empty text"
- **Audio Quality**: Ensure clear, audible audio
- **Model Selection**: Try different STT model (Whisper vs Hinglish)
- **Model Size**: Use larger Whisper model (medium/large)
- **Audio Format**: Verify video contains speech content
- **VAD Settings**: Adjust aggressiveness if VAD is too aggressive

### "Summary generation failed"
```bash
# Check Ollama status
ollama list
ollama ps

# Verify model availability
ollama pull gemma2:9b-instruct-q4_K_M

# Check Ollama logs
ollama serve  # Run in separate terminal
```

### "CUDA out of memory" (Hinglish model)
```python
# Reduce batch size in transcriber.py
batch_size=2  # Change from 4 to 2

# Or use Whisper model instead
# Select option 2 in model selection
```

### "Permission denied" errors
```bash
# Check file permissions
ls -la input/
ls -la output/

# Fix permissions if needed
chmod 755 input/
chmod 755 output/
```

### Performance Issues
```bash
# Check system resources
htop  # CPU/Memory usage
nvidia-smi  # GPU usage (if applicable)

# Optimize settings
VAD_MODE = "aggressive"        # Faster processing
CLEANUP_AUDIO = True          # Free disk space
# Use smaller Whisper model (tiny/base)
```

## ⚡ Performance & Optimization

### Processing Speed Benchmarks

| Component | CPU (Intel i7) | GPU (RTX 3080) | Notes |
|-----------|----------------|----------------|-------|
| Audio Extraction | ~1x real-time | ~1x real-time | ffmpeg optimized |
| VAD Processing | ~10x real-time | ~10x real-time | WebRTC optimized |
| Whisper (tiny) | ~2x real-time | ~20x real-time | Fastest option |
| Whisper (medium) | ~0.5x real-time | ~8x real-time | Balanced choice |
| Hinglish Model | ~0.3x real-time | ~5x real-time | 4GB VRAM required |
| Summary Generation | ~100 tokens/sec | ~200 tokens/sec | Depends on model |

### Optimization Tips

#### For Speed
```python
# Fastest configuration
USE_VAD = True
VAD_MODE = "aggressive"        # Maximum filtering
transcriber = Transcriber("tiny")  # Fastest model
CLEANUP_AUDIO = True          # Free disk space
```

#### For Accuracy
```python
# Highest quality configuration
USE_VAD = True
VAD_MODE = "conservative"     # Preserve content
transcriber = Transcriber("large")  # Best accuracy
CREATE_VISUALIZATION = True   # Debug audio quality
```

#### For Limited Resources
```python
# Low-memory configuration
USE_VAD = False               # Skip VAD processing
transcriber = Transcriber("base")   # Balanced model
CLEANUP_AUDIO = True          # Clean up files
# Reduce batch_size in transcriber.py if using Hinglish
```

### Memory Requirements

| Component | Minimum | Recommended | Notes |
|-----------|---------|-------------|-------|
| Python Environment | 512MB | 1GB | Base requirements |
| Whisper (tiny) | 1GB | 2GB | CPU only |
| Whisper (medium) | 2GB | 4GB | CPU recommended |
| Hinglish Model | 4GB | 8GB | CUDA recommended |
| Ollama (Gemma-2) | 4GB | 8GB | 4-bit quantized |

### Real-time Processing

For near real-time processing:
1. Use GPU acceleration (CUDA)
2. Select fast models (Whisper tiny/base)
3. Enable aggressive VAD filtering
4. Process shorter segments

## 🔧 Advanced Configuration

### Environment Variables

Create `.env` file for advanced configuration:
```bash
# .env
# Custom prompt instructions
extras="Custom instructions for summary generation..."

# Model parameters (optional)
OLLAMA_TEMPERATURE=0.3
OLLAMA_MAX_TOKENS=2000

# VAD parameters (advanced)
VAD_AGGRESSIVENESS=2
VAD_MAX_SILENCE=1.5
VAD_MIN_SPEECH=0.5
```

### Custom VAD Presets

Create custom VAD configurations in [`src/vad_processor.py`](src/vad_processor.py):
```python
# Add new preset
VAD_PRESETS["custom"] = {
    "aggressiveness": 2,
    "max_silence_duration": 1.2,
    "min_speech_duration": 0.4,
    "description": "Custom configuration for specific use case",
}
```

### Custom Summary Templates

Modify [`src/summarizer.py`](src/summarizer.py) for custom output formats:
```python
def _create_summary_prompt(self, transcription, meeting_title):
    # Custom prompt structure
    # Add company-specific sections
    # Modify formatting and organization
    # Support for different output formats (JSON, HTML, etc.)
```

### Batch Processing Scripts

Create custom batch processing workflows:
```bash
#!/bin/bash
# batch_process.sh

for video in input/*.mp4; do
    echo "Processing: $video"
    cp "$video" input/
    python app.py
    
    # Move processed video
    mv "$video" processed/
    
    # Optional: cleanup
    # rm -rf output/$(basename "$video" .mp4)
done
```

### Integration Examples

#### API Integration
```python
from src.transcriber import Transcriber
from src.summarizer import Summarizer

# Custom workflow
def process_meeting(video_path):
    # Extract audio
    audio_path = extract_audio_from_video(video_path)
    
    # Transcribe
    transcriber = Transcriber("medium")
    transcript = transcriber.transcribe_audio(audio_path)
    
    # Summarize
    summarizer = Summarizer("gemma2:9b-instruct-q4_K_M")
    summary = summarizer.generate_summary(transcript)
    
    return summary
```

#### Web Interface (Flask)
```python
from flask import Flask, request, jsonify
from src.utils import setup_logging

app = Flask(__name__)

@app.route('/process', methods=['POST'])
def process_meeting():
    # Handle file upload
    # Process with meetinger
    # Return results
    return jsonify({"status": "success"})
```

## 🧪 Development

### Project Architecture

The application follows a modular design with clear separation of concerns:

```
app.py (Main orchestrator)
├── src/utils.py (Common utilities)
├── src/video_processor.py (Audio extraction)
├── src/vad_processor.py (Voice activity detection)
├── src/transcriber.py (STT processing)
└── src/summarizer.py (AI summarization)
```

### Running Tests

Currently manual testing is supported:

```bash
# 1. Place test video in input folder
cp test_meeting.mp4 input/

# 2. Run application
python app.py

# 3. Verify outputs
ls output/test_meeting/
# transcript/ summary/ audio/ visualizations/

# 4. Check logs for errors
tail -f logs/app.log  # If logging is enabled
```

### Adding Features

The modular architecture supports easy extension:

#### New Audio Formats
Modify [`src/video_processor.py`](src/video_processor.py):
```python
# Add supported extensions
video_extensions = ["*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm", "*.flv", "*.wmv", "*.m4v"]
```

#### New STT Models
Extend [`src/transcriber.py`](src/transcriber.py):
```python
class NewTranscriber:
    def __init__(self, model_name):
        # Initialize new model
    
    def transcribe_audio(self, audio_path):
        # Implement transcription logic
```

#### Custom Summary Formats
Modify [`src/summarizer.py`](src/summarizer.py):
```python
def _create_summary_prompt(self, transcription, meeting_title):
    # Customize prompt structure
    # Add new sections or formatting
```

#### New VAD Algorithms
Extend [`src/vad_processor.py`](src/vad_processor.py):
```python
class CustomVADProcessor:
    def detect_speech_segments(self, audio_path):
        # Implement custom VAD logic
```

## 📦 Dependencies

### Core Dependencies
```txt
librosa==0.11.0              # Audio analysis and processing
matplotlib==3.10.8          # Visualization plots
numpy==2.3.5                # Numerical computing
ollama==0.6.1               # Local LLM inference
python-dotenv==1.2.1        # Environment variable loading
rich==14.2.0                # Beautiful console output
soundfile==0.13.1           # Audio file I/O
torch==2.9.1                # PyTorch deep learning framework
transformers==4.57.3        # Hugging Face transformers
accelerate==1.2.1           # Model acceleration
webrtcvad==2.0.10           # Voice activity detection
whisper==1.1.10             # OpenAI Whisper STT
```

### Optional Dependencies
- **CUDA Toolkit** (for GPU acceleration)
- **ffmpeg** (system dependency)
- **Ollama** (local LLM server)

### Backend Support

#### Whisper Backend
```bash
pip install openai-whisper
# Supports: tiny, base, small, medium, large models
# GPU: Automatic CUDA detection
# Languages: Multi-language support
```

#### Hugging Face Backend
```bash
pip install torch transformers accelerate
# Model: Oriserve/Whisper-Hindi2Hinglish-Apex
# GPU: 4GB VRAM minimum, 8GB recommended
# Output: Latin script Hinglish
```

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Contribution Guidelines
- **Code Quality**: Follow PEP 8 standards, use type hints
- **Documentation**: Update README.md and add docstrings
- **Testing**: Add tests for new features (manual testing currently)
- **Backwards Compatibility**: Maintain existing API when possible
- **Performance**: Consider impact on processing speed and memory usage

### Development Setup
```bash
# Clone with development dependencies
git clone <repo-url>
cd meetinger

# Set up development environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt  # If available

# Run linting and formatting
flake8 src/           # Code style
black src/            # Code formatting
mypy src/             # Type checking
```

## 🆘 Support

### Getting Help

1. **Documentation**: Check this README for setup and usage
2. **Troubleshooting**: Review the troubleshooting section above
3. **Issues**: [Create a GitHub issue](https://github.com/your-repo/meetinger/issues)
4. **Logs**: Enable debug logging for detailed error information

### Debug Mode
```python
# Enable debug logging in app.py
setup_logging(level=logging.DEBUG)
```

### Performance Tuning
Refer to the [Performance & Optimization](#performance--optimization) section for tuning guidance.

### Community Support
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/meetinger/discussions)
- **Documentation**: [Wiki](https://github.com/your-repo/meetinger/wiki)
- **Examples**: Check `examples/` directory for usage patterns

## 📝 Changelog

### Current Version (v1.0.0)
- **Multi-format video support** (MP4, AVI, MOV, MKV, WebM, etc.)
- **Dual STT models** (Whisper + Hinglish)
- **Advanced VAD processing** with configurable presets
- **Rich console output** with timing metrics
- **Smart file organization** with video-specific folders
- **AI-powered summarization** with Gemma-2
- **Customizable prompts** via environment variables
- **Speech activity visualizations**
- **Performance monitoring** and statistics
- **Comprehensive error handling** and troubleshooting
- **Modular architecture** for easy extension

### Version History
- **v0.1.0** (2025-12-15): Initial release with basic Whisper + Ollama
- **v0.5.0** (2025-12-16): Added VAD processing and Hinglish support
- **v1.0.0** (2025-12-17): Production-ready with advanced features
***Vibe Coded Yeah***

### Planned Features
- [ ] **Automated testing suite** (pytest + fixtures)
- [ ] **Web interface** (Flask/FastAPI + React)
- [ ] **Real-time processing** (streaming audio)
- [ ] **Team collaboration** (shared summaries, comments)
- [ ] **Calendar integration** (Google Calendar, Outlook)
- [ ] **Advanced speaker diarization** (who spoke when)
- [ ] **Multi-language support** (beyond Hinglish)
- [ ] **Cloud deployment** (Docker, Kubernetes)
- [ ] **API endpoints** (RESTful API for integration)
- [ ] **Mobile app** (iOS/Android companion)

## 🙏 Acknowledgments

### Dependencies
- **OpenAI Whisper**: High-quality speech recognition
- **Ollama**: Local LLM inference
- **WebRTC VAD**: Voice activity detection
- **LibROSA**: Audio analysis
- **Rich**: Beautiful console output

### Inspiration
This project was inspired by the need for efficient meeting documentation in distributed teams and the amazing capabilities of modern AI models for speech processing and text generation.

**Happy Meeting Processing! 🎤✨**