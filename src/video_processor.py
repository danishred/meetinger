"""
Video processing module for extracting audio from MP4 files.

This module provides functionality to extract audio from video files
using ffmpeg, with support for various video formats and audio configurations.
"""

import logging
from pathlib import Path
from typing import Optional, Union
import subprocess


def extract_audio_from_video(
    video_path: Path, output_dir: Union[str, Path] = "output"
) -> Optional[Path]:
    """
    Extract audio from video file using ffmpeg.

    Args:
        video_path: Path to the input video file
        output_dir: Directory to save the extracted audio (can be string or Path)

    Returns:
        Path to the extracted audio file (WAV format), or None if extraction failed
    """
    try:
        # Validate input file
        if not video_path.exists():
            logging.error(f"Video file not found: {video_path}")
            return None

        if not video_path.is_file():
            logging.error(f"Path is not a file: {video_path}")
            return None

        # Use the provided output directory (which should be video-specific)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate output audio filename
        audio_filename = video_path.stem + ".wav"
        audio_path = output_path / audio_filename

        logging.info(f"Extracting audio from {video_path.name}...")

        # Use ffmpeg to extract audio
        # -i: input file
        # -vn: disable video recording
        # -acodec pcm_s16le: PCM 16-bit little-endian audio codec
        # -ar 16000: sample rate 16kHz (good for speech recognition)
        # -ac 1: mono audio
        # -y: Overwrite output file if it exists
        command = [
            "ffmpeg",
            "-i",
            str(video_path),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-y",  # Overwrite output file if it exists
            str(audio_path),
        ]

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            check=False,  # Don't raise exception on non-zero return code
        )

        # Check if ffmpeg succeeded
        if result.returncode != 0:
            logging.error(f"ffmpeg failed with return code {result.returncode}")
            if result.stderr:
                logging.error(f"ffmpeg stderr: {result.stderr}")
            return None

        # Verify output file was created and has content
        if not audio_path.exists():
            logging.error(f"Audio file was not created: {audio_path}")
            return None

        # Check file size to ensure extraction was successful
        file_size = audio_path.stat().st_size
        if file_size == 0:
            logging.error(f"Extracted audio file is empty: {audio_path}")
            try:
                audio_path.unlink()  # Remove empty file
            except Exception as e:
                logging.warning(f"Failed to remove empty audio file: {e}")
            return None

        logging.info(
            f"Audio extracted successfully: {audio_path.name} ({file_size / 1024 / 1024:.2f} MB)"
        )
        return audio_path

    except subprocess.TimeoutExpired:
        logging.error(f"Audio extraction timed out after 5 minutes")
        return None
    except PermissionError as e:
        logging.error(f"Permission denied accessing file {video_path}: {e}")
        return None
    except Exception as e:
        logging.error(f"Failed to extract audio from {video_path}: {e}")
        return None


def cleanup_audio_file(audio_path: Path) -> bool:
    """
    Remove temporary audio file.

    Args:
        audio_path: Path to the audio file to remove

    Returns:
        True if file was removed successfully, False otherwise
    """
    try:
        if audio_path.exists():
            if not audio_path.is_file():
                logging.warning(f"Path is not a file: {audio_path}")
                return False

            audio_path.unlink()
            logging.info(f"Cleaned up temporary audio file: {audio_path.name}")
        return True
    except PermissionError as e:
        logging.warning(f"Permission denied removing {audio_path}: {e}")
        return False
    except FileNotFoundError:
        # File already removed, which is fine
        logging.debug(f"Audio file already removed: {audio_path}")
        return True
    except Exception as e:
        logging.warning(f"Failed to remove audio file {audio_path}: {e}")
        return False
