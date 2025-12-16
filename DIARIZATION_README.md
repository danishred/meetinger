# Speaker Diarization Feature

This document describes the speaker diarization feature added to the meeting transcription system.

## Overview

Speaker diarization identifies "who spoke when" in an audio recording. This feature enhances meeting transcripts by labeling each speech segment with the corresponding speaker, making it easier to follow conversations and attribute statements to specific participants.

## Features

- **Automatic Speaker Identification**: Detects and labels different speakers in the conversation
- **Enhanced Markdown Output**: Creates transcripts with speaker labels and timestamps
- **Speaker Statistics**: Provides summary statistics about each speaker's participation
- **Fallback Handling**: Gracefully handles failures and falls back to regular transcription
- **GPU Memory Management**: Automatically manages VRAM usage and falls back to CPU if needed
- **Compatible with Both Models**: Works with both Whisper and Whisper-Hinglish models

## Installation

### Prerequisites

1. Ensure you have the base requirements installed:
```bash
pip install -r requirements.txt
```

2. The speaker diarization feature requires additional dependencies:
```bash
# Core dependencies (should already be installed)
pip install torch>=2.0.0

# Pyannote audio dependencies
pip install pyannote.audio @ git+https://github.com/pyannote/pyannote-audio.git
pip install pyannote.core
pip install pyannote.database
pip install pyannote.metrics
pip install librosa>=0.9.0
pip install soundfile>=0.10.0
```

### Hugging Face Authentication

The pyannote speaker diarization models require a Hugging Face authentication token:

1. Create an account at [huggingface.co](https://huggingface.co)
2. Generate an access token in your account settings
3. Set the token as an environment variable:
```bash
export HF_TOKEN="your_token_here"
```

Or pass it to the Diarizer constructor:
```python
from src.diarizer import Diarizer
diarizer = Diarizer(use_auth_token="your_token_here")
```

## Usage

### Interactive Mode

When running `app.py`, you'll be prompted to enable speaker diarization:

```bash
python app.py
```

You'll see this prompt:
```
============================================================
Speaker Diarization:
============================================================
Enable speaker diarization to identify who spoke when?
Note: This requires additional processing time and GPU memory
============================================================
Enable speaker diarization? (y/n) [n]: 
```

Enter `y` or `yes` to enable diarization.

### Programmatic Usage

```python
from pathlib import Path
from src.diarizer import Diarizer
from src.transcriber import Transcriber

# Initialize components
transcriber = Transcriber(model_size="medium")
diarizer = Diarizer()

# Load models
transcriber.load_model()
diarizer.load_model()

# Process audio
audio_path = Path("path/to/audio.wav")

# Transcribe with timestamps
transcription_result = transcriber.transcribe_with_timestamps(audio_path)

# Perform diarization
diarization_result = diarizer.diarize(audio_path)

# Merge results
merged_segments = diarizer.merge_with_transcription(
    diarization_result, 
    transcription_result
)

# Save diarized transcript
diarizer.save_diarized_transcript(
    merged_segments, 
    video_base_name="meeting",
    title="Meeting Transcript"
)

# Get speaker statistics
speaker_summary = diarizer.get_speaker_summary(diarization_result)
print(f"Total speakers: {speaker_summary['total_speakers']}")
```

## Output Format

### Standard Transcript (Without Diarization)
```markdown
# Meeting Transcript

Hello everyone, welcome to today's meeting. Let's start with the agenda items.
Thank you for the introduction. I'd like to discuss the budget first.
```

### Diarized Transcript
```markdown
# Meeting Transcript

**[00:00:00 - SPEAKER_00]** Hello everyone, welcome to today's meeting. Let's start with the agenda items.

**[00:00:05 - SPEAKER_01]** Thank you for the introduction. I'd like to discuss the budget first.

**[00:00:12 - SPEAKER_00]** Great, let's begin with item number one.
```

## Speaker Statistics

When diarization is enabled, the system displays speaker participation statistics:

```
👥 Speaker Summary:
   SPEAKER_00: 125.4s across 23 segments
   SPEAKER_01: 89.2s across 18 segments
   SPEAKER_02: 45.6s across 12 segments
```

## File Output

- **Standard transcript**: `transcript/{video_name}_transcript.md`
- **Diarized transcript**: `transcript/{video_name}_diarized_transcript.md`

## Performance Considerations

### Processing Time
- Diarization adds approximately 2-5 minutes of processing time (depending on audio length)
- Total processing time is displayed in the timing summary

### Memory Requirements
- **Minimum GPU Memory**: 4GB VRAM recommended
- **Automatic Fallback**: If GPU memory is insufficient, automatically falls back to CPU
- **Memory Management**: Automatically clears CUDA cache after processing

### GPU vs CPU
- **GPU**: Significantly faster (recommended)
- **CPU**: Slower but works if GPU is unavailable or memory is limited

## Error Handling

The diarization feature includes comprehensive error handling:

1. **Model Loading Failure**: Falls back to regular transcription
2. **Diarization Failure**: Continues without speaker labels
3. **Memory Issues**: Automatically switches to CPU mode
4. **Missing Dependencies**: Provides clear error messages

All errors are logged and displayed to the user with appropriate fallback behavior.

## Troubleshooting

### Common Issues

1. **"ModuleNotFoundError: No module named 'pyannote'"**
   - Solution: Install pyannote dependencies (see Installation)

2. **"HF_TOKEN not found"**
   - Solution: Set Hugging Face authentication token (see Hugging Face Authentication)

3. **"CUDA out of memory"**
   - Solution: System automatically falls back to CPU mode
   - Alternative: Close other GPU-intensive applications

4. **"Diarization model failed to load"**
   - Solution: Check internet connection and HF_TOKEN
   - System will continue with regular transcription

5. **Slow performance**
   - Solution: Ensure you're using GPU if available
   - Consider using smaller audio segments for very long meetings

## Testing

Run the diarization test suite:

```bash
python test_diarization.py
```

This tests:
- Diarizer initialization
- Model loading
- Markdown formatting
- Speaker summary generation

## Architecture

### Components

1. **Diarizer Class** (`src/diarizer.py`)
   - `__init__()`: Initialize with model configuration
   - `load_model()`: Load pyannote.audio pipeline
   - `diarize()`: Perform speaker diarization
   - `merge_with_transcription()`: Combine speaker and text data
   - `format_diarized_markdown()`: Format output with speaker labels
   - `save_diarized_transcript()`: Save to file
   - `get_speaker_summary()`: Generate statistics

2. **Integration** (`app.py`)
   - Optional diarization step in processing pipeline
   - User prompt to enable/disable feature
   - Fallback handling for failures
   - Speaker statistics display

3. **Dependency Checking** (`src/utils.py`)
   - Validates pyannote.audio dependencies
   - Provides clear installation instructions

## Model Information

- **Model**: `pyannote/speaker-diarization-3.1`
- **Framework**: PyTorch
- **Input**: Audio file (WAV, MP3, etc.)
- **Output**: Speaker segments with timestamps

## Limitations

1. **Accuracy**: Speaker diarization accuracy depends on audio quality
2. **Overlapping Speech**: May not correctly identify overlapping speakers
3. **Short Segments**: Very short speech segments may not be detected
4. **Background Noise**: High background noise can reduce accuracy
5. **Processing Time**: Adds significant processing time to the workflow

## Future Enhancements

Potential improvements for future versions:

1. **Custom Speaker Labels**: Allow users to assign custom names to speakers
2. **Speaker Enrollment**: Train on known speaker voices for better accuracy
3. **Real-time Processing**: Enable live diarization during recording
4. **Multi-language Support**: Optimize for different languages
5. **Confidence Scores**: Display confidence levels for speaker assignments

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the logs for detailed error messages
3. Ensure all dependencies are properly installed
4. Verify Hugging Face authentication token is set correctly