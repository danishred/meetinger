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


def get_most_recent_mp4(video_dir: str) -> Optional[Path]:
    """
    Get the most recently modified MP4 file in the specified directory.

    Args:
        video_dir: Directory to search for MP4 files

    Returns:
        Path to the most recent MP4 file, or None if no MP4 files found
    """
    video_path = Path(video_dir)

    if not video_path.exists():
        logging.error(f"Video directory does not exist: {video_dir}")
        return None

    mp4_files = list(video_path.glob("*.mp4"))

    if not mp4_files:
        logging.warning(f"No MP4 files found in {video_dir}")
        return None

    # Sort by modification time and return the most recent
    most_recent = max(mp4_files, key=lambda p: p.stat().st_mtime)
    logging.info(f"Found most recent MP4: {most_recent.name}")

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

    Returns:
        True if all dependencies are available, False otherwise
    """
    dependencies_ok = True

    # Check for ffmpeg
    try:
        import subprocess

        result = subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            logging.error("ffmpeg is not installed or not working properly")
            dependencies_ok = False
        else:
            logging.info("ffmpeg is available")
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logging.error(f"ffmpeg not found: {e}")
        dependencies_ok = False

    # Check for Whisper
    try:
        import whisper

        logging.info("Whisper is available")
    except ImportError as e:
        logging.error(f"Whisper not installed: {e}")
        dependencies_ok = False

    # Check for Ollama
    try:
        import ollama

        # Try to connect to Ollama server
        try:
            ollama.list()
            logging.info("Ollama is available and server is running")
        except Exception as e:
            logging.error(f"Ollama server not accessible: {e}")
            dependencies_ok = False
    except ImportError as e:
        logging.error(f"Ollama not installed: {e}")
        dependencies_ok = False

    # Check for moviepy
    try:
        import moviepy

        logging.info("MoviePy is available")
    except ImportError as e:
        logging.error(f"MoviePy not installed: {e}")
        dependencies_ok = False

    return dependencies_ok
