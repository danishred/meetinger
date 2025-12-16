# Meeting Summary Generator

A Python application that automatically generates meeting summaries from MP4 video files. The application extracts audio, transcribes it using OpenAI's Whisper model, optionally performs speaker diarization to identify who spoke when, and generates structured markdown summaries using Ollama.

## Features

- **Automatic Processing**: Simply drop an MP4 file in the `videos/` folder and run the application
- **Local Transcription**: Uses OpenAI's Whisper or Whisper-Hinglish models for accurate speech-to-text
- **Speaker Diarization**: Optional feature to identify who spoke when (requires additional setup)
- **AI-Powered Summaries**: Generates structured markdown summaries using Ollama
- **Clean Workflow**: Minimal setup, maximum automation
- **Progress Logging**: Real-time feedback on processing steps
- **Error Handling**: Comprehensive error checking and user-friendly messages
- **Flexible Models**: Supports both English and Hinglish transcription

## System Architecture

```
MP4 File → Audio Extraction (ffmpeg) → Transcription (Whisper/Whisper-Hinglish) → [Optional: Speaker Diarization] → Transcript File (Markdown) → Summary (Ollama) → Summary Markdown
```

## Prerequisites

- **Python 3.8 or higher**
- **ffmpeg** - For audio extraction
- **Ollama** - For running LLM models locally

### Installing Prerequisites

#### Python
Ensure you have Python 3.8+ installed:
```bash
python3 --version
```

#### ffmpeg
**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**macOS (using Homebrew):**
```bash
brew install ffmpeg
```

**Windows:**
Download from [ffmpeg.org](https://ffmpeg.org/download.html) or use Chocolatey:
```powershell
choco install ffmpeg
```

#### Ollama
Download and install from [ollama.ai](https://ollama.ai/)

After installation, start the Ollama service:
```bash
ollama serve
```

## Installation

1. **Clone or download this repository**

2. **Create a virtual environment (recommended):**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Python dependencies:**
```bash
pip install -r requirements.txt
```

4. **Pull the required Ollama model:**
```bash
ollama pull qwen2.5:7b
```

### Optional: Speaker Diarization Setup

To enable speaker diarization (identifying who spoke when):

1. **Install additional dependencies:**
```bash
pip install pyannote.audio @ git+https://github.com/pyannote/pyannote-audio.git
pip install pyannote.core pyannote.database pyannote.metrics
pip install librosa>=0.9.0 soundfile>=0.10.0
```

2. **Set up Hugging Face authentication:**
   - Create an account at [huggingface.co](https://huggingface.co)
   - Generate an access token in your account settings
   - Set the token: `export HF_TOKEN="your_token_here"`

See [`DIARIZATION_README.md`](DIARIZATION_README.md) for detailed setup instructions.

## Project Structure

```
meeting-summary-generator/
├── app.py                  # Main application entry point
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── .gitignore             # Git ignore rules
├── videos/                # Input directory for MP4 files
├── transcript/            # Directory for saved transcript files
├── output/                # Output directory for summaries
└── src/                   # Source code modules
    ├── utils.py           # Utility functions
    ├── video_processor.py # Video to audio conversion
    ├── transcriber.py     # Whisper transcription
    ├── diarizer.py        # Speaker diarization (optional)
    └── summarizer.py      # Ollama summary generation
```

## Usage

### Basic Usage

1. **Place your MP4 meeting recording in the `videos/` folder**

2. **Run the application:**
```bash
python app.py
```

3. **Find your files:**
   - **Transcript**: `transcript/{video_name}_transcript.md`
   - **Diarized Transcript** (if enabled): `transcript/{video_name}_diarized_transcript.md`
   - **Summary**: `output/{video_name}_summary.md`

### Example

```bash
# Place your meeting recording
cp my_meeting.mp4 videos/

# Run the application
python app.py

# Check the output
ls output/
# my_meeting.md
```

### Processing Workflow

The application follows this 7-step process:

1. **Dependency Check**: Verifies all required tools are installed
2. **Find Video**: Locates the most recently modified video file
3. **Prepare Output**: Ensures output directory exists
4. **Extract Audio**: Uses ffmpeg to extract audio (WAV format, 16kHz mono)
5. **Transcribe**: Uses Whisper or Whisper-Hinglish model to convert audio to text and saves transcript to `transcript/` folder
6. **Speaker Diarization** (optional): Identifies speakers and creates diarized transcript with speaker labels
7. **Generate Summary**: Reads transcript from file and uses Ollama to create structured markdown summary

### File Naming Convention

- **Video**: `input/meeting.mp4` (or Windows Videos folder)
- **Transcript**: `transcript/meeting_transcript.md`
- **Diarized Transcript**: `transcript/meeting_diarized_transcript.md` (if diarization enabled)
- **Summary**: `output/meeting.md`

### Output Format

#### Standard Transcript
The transcript is saved as clean markdown with proper formatting.

#### Diarized Transcript (Optional)
When speaker diarization is enabled, the transcript includes speaker labels and timestamps:

```markdown
**[00:00:00 - SPEAKER_00]** Hello everyone, welcome to today's meeting.
**[00:00:05 - SPEAKER_01]** Thank you for the introduction.
```

#### Summary
The generated markdown summary includes:

- **Key Points**: Most important decisions and highlights
- **Action Items**: Checklist of tasks with assignees (if mentioned)
- **Discussion Topics**: Overview of topics covered
- **Decisions Made**: Important decisions documented
- **Next Steps**: Follow-up actions and future topics
- **Attendees**: Participants mentioned in the transcription

## Configuration

### Modifying Settings

Edit [`app.py`](app.py) to change configuration:

```python
# Configuration constants
VIDEO_DIR = "/mnt/c/Users/danis/Videos"  # Input directory (Windows Videos folder in WSL)
OUTPUT_DIR = "output"             # Output directory
OLLAMA_MODEL = "qwen2.5:7b"       # Ollama model name
CLEANUP_AUDIO = True              # Set False to keep intermediate audio files
ENABLE_DIARIZATION = False        # Set True to enable speaker diarization
```

### Available Transcription Models

#### Whisper Models (English)
- `tiny` - Fastest, lowest accuracy
- `base` - Fast, decent accuracy
- `small` - Balanced speed/accuracy
- `medium` - Good accuracy, moderate speed (default)
- `large` - Highest accuracy, slowest

#### Whisper-Hinglish Model
- Specialized for Hindi-English code-switching (Hinglish)
- Outputs text in Latin script
- Optimized for Indian English accents

### Alternative Ollama Models

You can use other models available in Ollama:

```bash
# Try different models
ollama pull qwen2.5:7b    # Current default
ollama pull llama2
ollama pull mistral
ollama pull gemma2:2b

# Then update the model name in app.py
OLLAMA_MODEL = "llama2"
```

## Troubleshooting

### "ffmpeg not found"
Ensure ffmpeg is installed and in your system PATH:
```bash
ffmpeg -version
```

### "Whisper not installed"
Reinstall the requirements:
```bash
pip install -r requirements.txt
```

### "Ollama server not accessible"
Ensure Ollama is running:
```bash
ollama serve
```

### "Model not found"
Pull the required model:
```bash
ollama pull google/gemma-3-1b
```

### "No MP4 files found"
Place your MP4 file in the `videos/` directory:
```bash
mkdir -p videos
cp your_meeting.mp4 videos/
```

### "Transcription returned empty text"
- Check if the audio is clear and audible
- Try a different Whisper model size (e.g., `medium` or `large`)
- Ensure the video file contains speech

### "Summary generation failed"
- Verify Ollama is running: `ollama list`
- Check if the model is available: `ollama ps`
- Try pulling the model again: `ollama pull qwen2.5:7b`

### "Diarization failed"
- Check Hugging Face token is set: `echo $HF_TOKEN`
- Verify pyannote dependencies are installed
- System will automatically fall back to regular transcription
- See [`DIARIZATION_README.md`](DIARIZATION_README.md) for detailed troubleshooting

## Performance Notes

- **Whisper Medium Model**: Processes ~1 minute of audio per second (CPU)
- **Audio Extraction**: Nearly instantaneous with ffmpeg
- **Speaker Diarization**: Adds 2-5 minutes (depending on audio length)
- **Summary Generation**: Depends on transcription length and model speed
- **Total Processing Time**: Approximately 1-2x real-time without diarization, 2-3x with diarization

For faster processing, use `tiny` or `base` Whisper models. For better accuracy, use `medium` or `large`.

## Advanced Usage

### Keeping Intermediate Files

Set `CLEANUP_AUDIO = False` in [`app.py`](app.py) to keep the extracted WAV audio file for debugging.

### Batch Processing

To process multiple videos, you can modify the application or run it multiple times (it automatically processes the most recent file).

### Custom Summary Prompts

Edit the `_create_summary_prompt()` method in [`src/summarizer.py`](src/summarizer.py) to customize the summary format.

### Using GPU for Whisper

For faster transcription, ensure you have CUDA-compatible GPU and install:
```bash
pip install openai-whisper
```
Whisper will automatically use GPU if available.

## Development

### Running Tests

Currently, no automated tests are included. Manual testing process:

1. Place a test MP4 file in `videos/`
2. Run `python app.py`
3. Verify the output markdown file in `output/`

### Adding Features

The modular architecture makes it easy to extend:

- **Add new audio formats**: Modify [`src/video_processor.py`](src/video_processor.py)
- **Change transcription model**: Modify [`src/transcriber.py`](src/transcriber.py)
- **Customize diarization**: Modify [`src/diarizer.py`](src/diarizer.py)
- **Customize summary format**: Modify [`src/summarizer.py`](src/summarizer.py)
- **Add new utilities**: Extend [`src/utils.py`](src/utils.py)

## Dependencies

### Core Dependencies
- **openai-whisper** (20231117): Speech-to-text transcription
- **ollama** (0.1.7): Local LLM inference
- **moviepy** (1.0.3): Video processing
- **torch** (>=2.0.0): Deep learning framework
- **transformers** (>=4.30.0): Hugging Face models
- **accelerate** (>=0.20.0): Model acceleration

### Optional: Speaker Diarization
- **pyannote.audio**: Speaker diarization pipeline
- **pyannote.core**: Core pyannote utilities
- **pyannote.database**: Database utilities
- **pyannote.metrics**: Evaluation metrics
- **librosa** (>=0.9.0): Audio processing
- **soundfile** (>=0.10.0): Audio file I/O

## License

This project is open source. Feel free to modify and distribute.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Support

For issues or questions:
1. Check the Troubleshooting section above
2. Review the logs output by the application
3. Ensure all prerequisites are properly installed

## Version History

- **v2.0.0** (2025-12-16): Speaker Diarization Update
  - Added optional speaker diarization feature
  - Support for Whisper-Hinglish model
  - Enhanced markdown output with speaker labels
  - Improved GPU memory management
  - Updated default Ollama model to qwen2.5:7b

- **v1.0.0** (2025-12-15): Initial release
  - MP4 to audio extraction
  - Whisper transcription
  - Ollama summarization
  - Markdown output format