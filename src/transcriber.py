"""
Transcription module using OpenAI's Whisper model.
"""

import logging
from pathlib import Path
from typing import Optional
import whisper


class Transcriber:
    """Handles audio transcription using Whisper."""

    def __init__(self, model_size: str = "medium"):
        """
        Initialize the transcriber with specified Whisper model.

        Args:
            model_size: Size of Whisper model to use (tiny, base, small, medium, large)
        """
        self.model_size = model_size
        self.model = None
        logging.info(f"Initializing Whisper model: {model_size}")

    def load_model(self) -> bool:
        """
        Load the Whisper model.

        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            self.model = whisper.load_model(self.model_size)
            logging.info(f"Whisper model '{self.model_size}' loaded successfully")
            return True
        except Exception as e:
            logging.error(f"Failed to load Whisper model: {e}")
            return False

    def save_transcript_to_file(
        self, transcript_text: str, video_base_name: str
    ) -> Optional[Path]:
        """
        Save transcript to a markdown file in the transcript folder.

        Args:
            transcript_text: The transcribed text to save
            video_base_name: Base name of the video file (without extension)

        Returns:
            Path to the saved transcript file, or None if save failed
        """
        try:
            # Create transcript directory if it doesn't exist
            transcript_dir = Path("transcript")
            transcript_dir.mkdir(parents=True, exist_ok=True)

            # Create transcript file path
            transcript_path = transcript_dir / f"{video_base_name}_transcript.md"

            # Format transcript as markdown with proper line breaks
            formatted_transcript = self._format_transcript_as_markdown(transcript_text)

            # Write to file
            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write(formatted_transcript)

            logging.info(f"Transcript saved to: {transcript_path}")
            return transcript_path

        except Exception as e:
            logging.error(f"Failed to save transcript to file: {e}")
            return None

    def _format_transcript_as_markdown(self, transcript_text: str) -> str:
        """
        Format transcript text as clean markdown with proper line breaks.

        Args:
            transcript_text: Raw transcript text

        Returns:
            Formatted markdown string
        """
        # Add markdown header
        formatted = "# Meeting Transcript\n\n"

        # Clean up the transcript text
        # Ensure proper spacing and line breaks
        cleaned_text = transcript_text.strip()

        # Add the transcript content
        formatted += cleaned_text

        # Ensure the file ends with a newline
        if not formatted.endswith("\n"):
            formatted += "\n"

        return formatted

    def transcribe_audio(
        self, audio_path: Path, video_base_name: str = None
    ) -> Optional[str]:
        """
        Transcribe audio file to text and save to transcript folder.

        Args:
            audio_path: Path to the audio file to transcribe
            video_base_name: Optional base name for saving transcript file

        Returns:
            Transcribed text, or None if transcription failed
        """
        if not audio_path.exists():
            logging.error(f"Audio file not found: {audio_path}")
            return None

        if self.model is None:
            logging.error("Whisper model not loaded")
            return None

        logging.info(f"Transcribing audio: {audio_path.name}")

        try:
            # Transcribe the audio
            result = self.model.transcribe(
                str(audio_path),
                language="en",  # Specify language for better accuracy
                task="transcribe",  # Transcribe (not translate)
                verbose=False,  # Suppress Whisper's progress output
            )

            transcription = result.get("text", "").strip()

            if not transcription:
                logging.warning("Transcription returned empty text")
                return None

            # Log some statistics
            segments = result.get("segments", [])
            if segments:
                duration = segments[-1]["end"] if segments else 0
                logging.info(
                    f"Transcription complete: {len(segments)} segments, {duration:.1f}s duration"
                )
                logging.info(
                    f"Transcribed text length: {len(transcription)} characters"
                )

            # Save transcript to file if video_base_name is provided
            if video_base_name:
                transcript_path = self.save_transcript_to_file(
                    transcription, video_base_name
                )
                if transcript_path:
                    logging.info(f"Transcript saved to: {transcript_path}")
                else:
                    logging.warning("Failed to save transcript to file")

            return transcription

        except Exception as e:
            logging.error(f"Transcription failed: {e}")
            return None

    def transcribe_with_timestamps(self, audio_path: Path) -> Optional[dict]:
        """
        Transcribe audio file with word-level timestamps.

        Args:
            audio_path: Path to the audio file to transcribe

        Returns:
            Dictionary with transcription and segment information, or None if failed
        """
        if not audio_path.exists():
            logging.error(f"Audio file not found: {audio_path}")
            return None

        if self.model is None:
            logging.error("Whisper model not loaded")
            return None

        logging.info(f"Transcribing audio with timestamps: {audio_path.name}")

        try:
            # Transcribe with word timestamps
            result = self.model.transcribe(
                str(audio_path),
                language="en",
                task="transcribe",
                verbose=False,
                word_timestamps=True,  # Include word-level timestamps
            )

            return result

        except Exception as e:
            logging.error(f"Transcription with timestamps failed: {e}")
            return None
