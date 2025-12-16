# Meeting Summary Generator

A Python application that automatically generates meeting summaries from MP4 video files. The application extracts audio, transcribes it using OpenAI's Whisper model, and generates structured markdown summaries using Ollama with Google's Gemma model.

## Features

- **Automatic Processing**: Simply drop an MP4 file in the `videos/` folder and run the application
- **Local Transcription**: Uses OpenAI's Whisper medium model for accurate speech-to-text
- **AI-Powered Summaries**: Generates structured markdown summaries using Google Gemma-3-1B via Ollama
- **Clean Workflow**: Minimal setup, maximum automation
- **Progress Logging**: Real-time feedback on processing steps
- **Error Handling**: Comprehensive error checking and user-friendly messages

## System Architecture

```
MP4 File → Audio Extraction (ffmpeg) → Transcription (Whisper) → Transcript File (Markdown) → Summary (Ollama/Gemma) → Summary Markdown
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
ollama pull google/gemma-3-1b
```

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

The application follows this 6-step process:

1. **Dependency Check**: Verifies all required tools are installed
2. **Find MP4**: Locates the most recently modified MP4 file in `videos/`
3. **Prepare Output**: Ensures output directory exists
4. **Extract Audio**: Uses ffmpeg to extract audio (WAV format, 16kHz mono)
5. **Transcribe**: Uses Whisper medium model to convert audio to text and saves transcript to `transcript/` folder
6. **Generate Summary**: Reads transcript from file and uses Ollama with Gemma-3-1B to create structured markdown summary

### File Naming Convention

- **Video**: `videos/meeting.mp4`
- **Transcript**: `transcript/meeting_transcript.md`
- **Summary**: `output/meeting_summary.md`

### Output Format

The transcript is saved as clean markdown with proper formatting.

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
# Configuration constants (lines 30-36)
VIDEO_DIR = "videos"              # Input directory
OUTPUT_DIR = "output"             # Output directory
WHISPER_MODEL = "medium"           # Whisper model size (tiny, base, small, medium, large)
OLLAMA_MODEL = "google/gemma-3-1b" # Ollama model name
CLEANUP_AUDIO = True              # Set False to keep intermediate audio files
```

### Available Whisper Models

- `tiny` - Fastest, lowest accuracy
- `base` - Fast, decent accuracy
- `medium` - Balanced speed/accuracy (default)
- `medium` - Slower, higher accuracy
- `large` - Slowest, highest accuracy

### Alternative Ollama Models

You can use other models available in Ollama:

```bash
# Try different models
ollama pull llama2
ollama pull codellama
ollama pull mistral

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
- Try pulling the model again: `ollama pull google/gemma-3-1b`

## Performance Notes

- **Whisper Medium Model**: Processes ~1 minute of audio per second (CPU)
- **Audio Extraction**: Nearly instantaneous with ffmpeg
- **Summary Generation**: Depends on transcription length and model speed
- **Total Processing Time**: Approximately 1-2x real-time (e.g., 30min meeting = 30-60min processing)

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
- **Customize summary format**: Modify [`src/summarizer.py`](src/summarizer.py)
- **Add new utilities**: Extend [`src/utils.py`](src/utils.py)

## Dependencies

- **openai-whisper** (20231117): Speech-to-text transcription
- **ollama** (0.1.7): Local LLM inference
- **moviepy** (1.0.3): Video processing (currently not used, reserved for future features)

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

- **v1.0.0** (2025-12-15): Initial release
  - MP4 to audio extraction
  - Whisper transcription
  - Ollama/Gemma summarization
  - Markdown output format