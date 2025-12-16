"""
Video processing module for extracting audio from MP4 files.
"""

import logging
from pathlib import Path
from typing import Optional
import subprocess


def extract_audio_from_video(
    video_path: Path, output_dir: str = "output"
) -> Optional[Path]:
    """
    Extract audio from video file using ffmpeg.

    Args:
        video_path: Path to the input video file
        output_dir: Directory to save the extracted audio

    Returns:
        Path to the extracted audio file (WAV format), or None if extraction failed
    """
    if not video_path.exists():
        logging.error(f"Video file not found: {video_path}")
        return None

    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate output audio filename
    audio_filename = video_path.stem + ".wav"
    audio_path = output_path / audio_filename

    logging.info(f"Extracting audio from {video_path.name}...")

    try:
        # Use ffmpeg to extract audio
        # -i: input file
        # -vn: disable video recording
        # -acodec pcm_s16le: PCM 16-bit little-endian audio codec
        # -ar 16000: sample rate 16kHz (good for speech recognition)
        # -ac 1: mono audio
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
            command, capture_output=True, text=True, timeout=300  # 5 minute timeout
        )

        if result.returncode != 0:
            logging.error(f"ffmpeg failed with return code {result.returncode}")
            logging.error(f"ffmpeg stderr: {result.stderr}")
            return None

        if not audio_path.exists():
            logging.error(f"Audio file was not created: {audio_path}")
            return None

        # Check file size to ensure extraction was successful
        file_size = audio_path.stat().st_size
        if file_size == 0:
            logging.error(f"Extracted audio file is empty: {audio_path}")
            audio_path.unlink()  # Remove empty file
            return None

        logging.info(
            f"Audio extracted successfully: {audio_path.name} ({file_size / 1024 / 1024:.2f} MB)"
        )
        return audio_path

    except subprocess.TimeoutExpired:
        logging.error(f"Audio extraction timed out after 5 minutes")
        return None
    except Exception as e:
        logging.error(f"Failed to extract audio: {e}")
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
            audio_path.unlink()
            logging.info(f"Cleaned up temporary audio file: {audio_path.name}")
        return True
    except Exception as e:
        logging.warning(f"Failed to remove audio file {audio_path}: {e}")
        return False
