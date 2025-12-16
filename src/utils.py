"""
Utility functions for file handling and logging.
"""

import os
import logging
from pathlib import Path
from typing import Optional, List


def setup_logging(level: int = logging.INFO) -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def get_most_recent_video(video_dir: str) -> Optional[Path]:
    """
    Get the most recently modified video file in the specified directory.
    Supports multiple video formats and handles Windows to WSL path conversion.

    Args:
        video_dir: Directory to search for video files (can be Windows path)

    Returns:
        Path to the most recent video file, or None if no video files found
    """
    # Convert Windows path to WSL path if needed
    if video_dir.startswith("C:\\"):
        # Convert Windows path like C:\Users\danis\Videos to WSL path
        # C:\Users\danis\Videos -> /mnt/c/Users/danis/Videos
        wsl_path = video_dir.replace("C:\\", "/mnt/c/").replace("\\", "/")
        video_path = Path(wsl_path)
        logging.info(f"Converted Windows path to WSL: {video_dir} -> {wsl_path}")
    else:
        video_path = Path(video_dir)

    if not video_path.exists():
        logging.error(f"Video directory does not exist: {video_path}")
        return None

    # Supported video file extensions
    video_extensions = [
        "*.mp4",
        "*.avi",
        "*.mov",
        "*.mkv",
        "*.webm",
        "*.flv",
        "*.wmv",
        "*.m4v",
    ]

    # Find all video files
    video_files = []
    for ext in video_extensions:
        video_files.extend(video_path.glob(ext))

    if not video_files:
        logging.warning(f"No video files found in {video_dir}")
        logging.warning(
            f"Supported formats: {', '.join([ext[2:] for ext in video_extensions])}"
        )
        return None

    # Sort by modification time and return the most recent
    most_recent = max(video_files, key=lambda p: p.stat().st_mtime)
    logging.info(f"Found most recent video: {most_recent.name}")

    return most_recent


def ensure_output_dir(output_dir: str) -> bool:
    """
    Ensure the output directory exists.

    Args:
        output_dir: Directory to create if it doesn't exist

    Returns:
        True if directory exists or was created successfully, False otherwise
    """
    output_path = Path(output_dir)

    try:
        output_path.mkdir(parents=True, exist_ok=True)
        logging.info(f"Output directory ready: {output_dir}")
        return True
    except Exception as e:
        logging.error(f"Failed to create output directory {output_dir}: {e}")
        return False


def construct_output_path(
    video_path: Path, output_dir: str, extension: str = ".md"
) -> Path:
    """
    Construct output file path based on video file name.

    Args:
        video_path: Path to the input video file
        output_dir: Directory for output files
        extension: File extension for output file (default: .md)

    Returns:
        Path for the output file
    """
    output_filename = video_path.stem + extension
    return Path(output_dir) / output_filename


def check_dependencies() -> bool:
    """
    Check if required dependencies are available.
    Supports both Whisper and Hugging Face backends for transcription.

    Returns:
        True if at least one transcription backend is available, False otherwise
    """
    import subprocess

    # Core dependencies (required for all backends)
    ffmpeg_available = False
    ollama_available = False
    moviepy_available = False

    # Transcription backend availability
    whisper_available = False
    pytorch_available = False
    transformers_available = False
    accelerate_available = False

    # Diarization backend availability
    pyannote_audio_available = False
    pyannote_core_available = False
    librosa_available = False
    soundfile_available = False

    # Check for ffmpeg (required for both backends)
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            logging.error("❌ ffmpeg is not installed or not working properly")
        else:
            logging.info("✅ ffmpeg is available")
            ffmpeg_available = True
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logging.error(f"❌ ffmpeg not found: {e}")

    # Check for Whisper
    try:
        import whisper

        logging.info("✅ Whisper library is available")
        whisper_available = True
    except ImportError:
        logging.warning("❌ Whisper library not found")

    # Check for PyTorch
    try:
        import torch

        logging.info("✅ PyTorch library is available")
        pytorch_available = True
    except ImportError:
        logging.warning("❌ PyTorch library not found")

    # Check for Transformers
    try:
        import transformers

        logging.info("✅ Transformers library is available")
        transformers_available = True
    except ImportError:
        logging.warning("❌ Transformers library not found")

    # Check for Accelerate
    try:
        import accelerate

        logging.info("✅ Accelerate library is available")
        accelerate_available = True
    except ImportError:
        logging.warning("❌ Accelerate library not found")

    # Check for Pyannote Audio
    try:
        import pyannote.audio

        logging.info("✅ Pyannote Audio library is available")
        pyannote_audio_available = True
    except ImportError:
        logging.warning("❌ Pyannote Audio library not found")

    # Check for Pyannote Core
    try:
        import pyannote.core

        logging.info("✅ Pyannote Core library is available")
        pyannote_core_available = True
    except ImportError:
        logging.warning("❌ Pyannote Core library not found")

    # Check for Librosa
    try:
        import librosa

        logging.info("✅ Librosa library is available")
        librosa_available = True
    except ImportError:
        logging.warning("❌ Librosa library not found")

    # Check for SoundFile
    try:
        import soundfile

        logging.info("✅ SoundFile library is available")
        soundfile_available = True
    except ImportError:
        logging.warning("❌ SoundFile library not found")

    # Check for Ollama (required for summarization)
    try:
        import ollama

        # Try to connect to Ollama server
        try:
            ollama.list()
            logging.info("✅ Ollama is available and server is running")
            ollama_available = True
        except Exception as e:
            logging.error(f"❌ Ollama server not accessible: {e}")
    except ImportError:
        logging.error("❌ Ollama not installed")

    # Check for moviepy (required for video processing)
    try:
        import moviepy

        logging.info("✅ MoviePy is available")
        moviepy_available = True
    except ImportError:
        logging.error("❌ MoviePy not installed")

    # Check if at least one transcription backend is available
    whisper_backend_available = ffmpeg_available and whisper_available
    huggingface_backend_available = (
        ffmpeg_available
        and pytorch_available
        and transformers_available
        and accelerate_available
    )

    # Check if diarization backend is available
    diarization_backend_available = (
        ffmpeg_available
        and pytorch_available
        and pyannote_audio_available
        and pyannote_core_available
        and librosa_available
        and soundfile_available
    )

    if not whisper_backend_available and not huggingface_backend_available:
        logging.error("❌ No transcription backend available!")
        logging.error("Install Whisper dependencies OR Hugging Face dependencies")
        logging.error("ffmpeg is required for both backends")
        return False

    # Log diarization backend availability
    if diarization_backend_available:
        logging.info("✅ Speaker diarization backend available")
    else:
        logging.warning("⚠️  Speaker diarization backend not fully available")
        logging.warning("   Install: pyannote.audio, pyannote.core, librosa, soundfile")

    # Check required core dependencies
    if not ollama_available:
        logging.error("❌ Ollama is required for summarization")
        return False

    if not moviepy_available:
        logging.error("❌ MoviePy is required for video processing")
        return False

    # Log which backends are available
    if whisper_backend_available:
        logging.info("✅ Whisper backend available")
    if huggingface_backend_available:
        logging.info("✅ Hugging Face backend available")

    logging.info("✅ All required dependencies are available")
    return True
