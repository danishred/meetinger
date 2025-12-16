"""
Utility functions for file handling and logging.

This module provides common utility functions for:
- Logging configuration
- File path management
- Video file discovery
- Output directory management
- Dependency checking
"""

import os
import logging
from pathlib import Path
from typing import Optional, List, Union


def setup_logging(level: int = logging.INFO) -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def get_most_recent_video(video_dir: Union[str, Path]) -> Optional[Path]:
    """
    Get the most recently modified video file in the specified directory.
    Supports multiple video formats and handles Windows to WSL path conversion.

    Args:
        video_dir: Directory to search for video files (can be Windows path or Path object)

    Returns:
        Path to the most recent video file, or None if no video files found

    Raises:
        ValueError: If the video directory path is invalid
    """
    try:
        # Convert Windows path to WSL path if needed
        if isinstance(video_dir, str) and video_dir.startswith("C:\\"):
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

        if not video_path.is_dir():
            logging.error(f"Path is not a directory: {video_path}")
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

    except Exception as e:
        logging.error(f"Error finding most recent video: {e}")
        return None


def ensure_output_dir(output_dir: Union[str, Path]) -> bool:
    """
    Ensure the output directory exists.

    Args:
        output_dir: Directory to create if it doesn't exist (can be string or Path)

    Returns:
        True if directory exists or was created successfully, False otherwise
    """
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        logging.info(f"Output directory ready: {output_path}")
        return True
    except PermissionError:
        logging.error(f"Permission denied: Cannot create directory {output_path}")
        return False
    except Exception as e:
        logging.error(f"Failed to create output directory {output_path}: {e}")
        return False


def construct_output_path(
    video_path: Path, output_dir: Union[str, Path], extension: str = ".md"
) -> Path:
    """
    Construct output file path based on video file name.

    Args:
        video_path: Path to the input video file
        output_dir: Directory for output files (can be string or Path)
        extension: File extension for output file (default: .md)

    Returns:
        Path for the output file
    """
    output_filename = video_path.stem + extension
    return Path(output_dir) / output_filename


def get_video_output_dir(
    video_path: Path, base_output_dir: Union[str, Path] = "output"
) -> Path:
    """
    Get the video-specific output directory path.

    Args:
        video_path: Path to the input video file
        base_output_dir: Base output directory (default: "output", can be string or Path)

    Returns:
        Path to the video-specific output directory
    """
    try:
        # Create directory name based on video filename (without extension)
        video_dir_name = video_path.stem
        video_output_dir = Path(base_output_dir) / video_dir_name

        # Ensure the directory exists
        video_output_dir.mkdir(parents=True, exist_ok=True)

        logging.debug(f"Video output directory: {video_output_dir}")
        return video_output_dir

    except Exception as e:
        logging.error(f"Failed to create video output directory: {e}")
        # Fall back to base output directory
        return Path(base_output_dir)


def check_dependencies() -> bool:
    """
    Check if required dependencies are available.
    Supports both Whisper and Hugging Face backends for transcription.

    Returns:
        True if at least one transcription backend is available, False otherwise
    """
    import subprocess

    # Core dependencies (required for all backends)
    dependencies = {
        "core": {
            "ffmpeg": False,
            "ollama": False,
            "moviepy": False,
        },
        "whisper": {
            "whisper": False,
        },
        "huggingface": {
            "torch": False,
            "transformers": False,
            "accelerate": False,
        },
    }

    # Check for ffmpeg (required for both backends)
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            logging.info("✅ ffmpeg is available")
            dependencies["core"]["ffmpeg"] = True
        else:
            logging.error("❌ ffmpeg is not installed or not working properly")
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logging.error(f"❌ ffmpeg not found: {e}")

    # Check for individual libraries
    _check_library_import("whisper", dependencies["whisper"])
    _check_library_import("torch", dependencies["huggingface"])
    _check_library_import("transformers", dependencies["huggingface"])
    _check_library_import("accelerate", dependencies["huggingface"])
    _check_library_import("moviepy", dependencies["core"])

    # Check for Ollama with server connectivity test
    _check_ollama_availability(dependencies["core"])

    # Determine backend availability
    whisper_backend_available = (
        dependencies["core"]["ffmpeg"] and dependencies["whisper"]["whisper"]
    )
    huggingface_backend_available = (
        dependencies["core"]["ffmpeg"]
        and dependencies["huggingface"]["torch"]
        and dependencies["huggingface"]["transformers"]
        and dependencies["huggingface"]["accelerate"]
    )

    # Check required core dependencies
    if not dependencies["core"]["ollama"]:
        logging.error("❌ Ollama is required for summarization")
        return False

    if not dependencies["core"]["moviepy"]:
        logging.error("❌ MoviePy is required for video processing")
        return False

    if not whisper_backend_available and not huggingface_backend_available:
        logging.error("❌ No transcription backend available!")
        logging.error("Install Whisper dependencies OR Hugging Face dependencies")
        logging.error("ffmpeg is required for both backends")
        return False

    # Log which backends are available
    if whisper_backend_available:
        logging.info("✅ Whisper backend available")
    if huggingface_backend_available:
        logging.info("✅ Hugging Face backend available")

    logging.info("✅ All required dependencies are available")
    return True


def _check_library_import(library_name: str, dependency_dict: dict) -> None:
    """
    Check if a library can be imported and update the dependency dictionary.

    Args:
        library_name: Name of the library to check
        dependency_dict: Dictionary to update with the result
    """
    try:
        __import__(library_name)
        logging.info(f"✅ {library_name} library is available")
        dependency_dict[library_name] = True
    except ImportError:
        logging.warning(f"❌ {library_name} library not found")
        dependency_dict[library_name] = False


def _check_ollama_availability(dependency_dict: dict) -> None:
    """
    Check if Ollama is available and server is running.

    Args:
        dependency_dict: Dictionary to update with the result
    """
    try:
        import ollama

        # Try to connect to Ollama server
        try:
            ollama.list()
            logging.info("✅ Ollama is available and server is running")
            dependency_dict["ollama"] = True
        except Exception as e:
            logging.error(f"❌ Ollama server not accessible: {e}")
            dependency_dict["ollama"] = False
    except ImportError:
        logging.error("❌ Ollama not installed")
        dependency_dict["ollama"] = False
