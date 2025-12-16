"""
Speaker diarization module using pyannote.audio.

This module provides speaker diarization capabilities to identify
who spoke when in audio recordings.
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Tuple

# Import with error handling for missing dependencies
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    logging.warning("PyTorch not available. Diarization will not work without it.")

try:
    from pyannote.audio import Pipeline

    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False
    Pipeline = None
    logging.warning(
        "pyannote.audio not available. Diarization will not work without it."
    )

# Type aliases for clarity
DiarizationResult = Dict
TranscriptionResult = Dict
MergedSegment = Dict


class Diarizer:
    """Handles speaker diarization using pyannote.audio pipeline."""

    def __init__(
        self,
        model_name: str = "pyannote/speaker-diarization-3.1",
        token: Optional[str] = None,
    ):
        """
        Initialize the diarizer with specified model.

        Args:
            model_name: Hugging Face model ID for speaker diarization
            token: Optional Hugging Face authentication token
        """
        self.model_name = model_name
        self.token = token
        self.pipeline = None

        # Check if dependencies are available
        if not TORCH_AVAILABLE:
            logging.error("PyTorch is not installed. Cannot initialize diarizer.")
            logging.error("Install with: pip install torch>=2.0.0")
            return

        if not PYANNOTE_AVAILABLE:
            logging.error(
                "pyannote.audio is not installed. Cannot initialize diarizer."
            )
            logging.error("Install with: pip install pyannote.audio")
            return

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Initializing Diarizer with model: {model_name}")
        logging.info(f"Using device: {self.device}")

    def load_model(self) -> bool:
        """
        Load the speaker diarization pipeline.

        Returns:
            True if model loaded successfully, False otherwise
        """
        # Check if dependencies are available
        if not TORCH_AVAILABLE or not PYANNOTE_AVAILABLE:
            logging.error("Cannot load model: Required dependencies are not installed.")
            logging.error("Install missing dependencies:")
            if not TORCH_AVAILABLE:
                logging.error("  - pip install torch>=2.0.0")
            if not PYANNOTE_AVAILABLE:
                logging.error("  - pip install pyannote.audio")
            return False

        logging.info(f"Starting model loading process for: {self.model_name}")
        logging.info(f"Using device: {self.device}")
        logging.info(f"Auth token configured: {'Yes' if self.token else 'No'}")

        # Set Hugging Face token in environment if provided
        if self.token:
            import os

            os.environ["HF_TOKEN"] = self.token
            logging.info("HF_TOKEN set in environment")

        try:
            # Check GPU memory before loading
            if torch.cuda.is_available():
                logging.info("CUDA is available, checking GPU memory...")
                torch.cuda.empty_cache()
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                logging.info(f"GPU memory available: {gpu_memory / 1024**3:.2f} GB")

                # Fallback to CPU if GPU memory is insufficient (< 4GB)
                if gpu_memory < 4 * 1024**3:
                    logging.warning(
                        "GPU memory may be insufficient for diarization. Falling back to CPU."
                    )
                    self.device = "cpu"
            else:
                logging.info("CUDA not available, will use CPU")

            # Load the pipeline
            logging.info(f"Loading pipeline from pretrained: {self.model_name}")
            logging.info(f"Using token: {self.token}")

            # Try to load with the specified model
            try:
                self.pipeline = Pipeline.from_pretrained(
                    self.model_name,
                    token=self.token,
                )
            except Exception as auth_error:
                # If authentication fails, log the error and suggest solutions
                if (
                    "401" in str(auth_error)
                    or "403" in str(auth_error)
                    or "gated" in str(auth_error).lower()
                ):
                    logging.warning(
                        f"Authentication failed for {self.model_name}: {auth_error}"
                    )
                    logging.warning("To use speaker diarization, you need to:")
                    logging.warning(
                        "1. Visit https://huggingface.co/pyannote/speaker-diarization-3.1"
                    )
                    logging.warning("2. Click 'Access repository' and accept the terms")
                    logging.warning(
                        "3. Visit https://huggingface.co/pyannote/segmentation-3.0"
                    )
                    logging.warning("4. Click 'Access repository' and accept the terms")
                    logging.warning("5. Ensure HF_TOKEN is set in your .env file")
                    # Re-raise the error since there's no working fallback
                    raise auth_error
                else:
                    # Re-raise if it's not an authentication error
                    raise auth_error

            logging.info("Pipeline loaded successfully, moving to device...")

            # Move pipeline to appropriate device
            self.pipeline.to(torch.device(self.device))

            logging.info(
                f"Diarization model '{self.model_name}' loaded successfully on {self.device}"
            )
            return True

        except Exception as e:
            logging.error(f"Failed to load diarization model: {e}")
            logging.error(f"Exception type: {type(e).__name__}")
            import traceback

            logging.error(f"Full traceback:\n{traceback.format_exc()}")
            self.pipeline = None
            return False

    def diarize(self, audio_path: Path) -> Optional[DiarizationResult]:
        """
        Perform speaker diarization on audio file.

        Args:
            audio_path: Path to the audio file to diarize

        Returns:
            Dictionary containing speaker segments with start/end times and speaker IDs,
            or None if diarization failed
        """
        logging.info(f"Starting diarization for: {audio_path}")

        if not audio_path.exists():
            logging.error(f"Audio file not found: {audio_path}")
            return None

        if self.pipeline is None:
            logging.error("Diarization model not loaded - pipeline is None")
            return None

        logging.info(f"Performing speaker diarization: {audio_path.name}")
        logging.info(f"Pipeline ready: {self.pipeline is not None}")
        logging.info(f"Using device: {self.device}")

        try:
            logging.info("Applying pipeline to audio file...")
            # Apply the pipeline to the audio file
            diarization = self.pipeline(str(audio_path))
            logging.info("Pipeline applied successfully")

            # Convert pyannote.core.Annotation to list of segments
            segments = []
            logging.info("Converting diarization results to segments...")
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append(
                    {
                        "start": turn.start,
                        "end": turn.end,
                        "speaker": speaker,
                        "duration": turn.end - turn.start,
                    }
                )

            logging.info(f"Collected {len(segments)} segments")

            # Sort segments by start time
            segments.sort(key=lambda x: x["start"])

            result = {
                "segments": segments,
                "num_speakers": len(set(seg["speaker"] for seg in segments)),
                "total_duration": segments[-1]["end"] if segments else 0,
            }

            logging.info(
                f"Diarization complete: {len(segments)} segments, "
                f"{result['num_speakers']} speakers, "
                f"{result['total_duration']:.1f}s duration"
            )

            # Clear CUDA cache after diarization
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logging.info("CUDA cache cleared")

            return result

        except Exception as e:
            logging.error(f"Diarization failed: {e}")
            logging.error(f"Exception type: {type(e).__name__}")
            import traceback

            logging.error(f"Full traceback:\n{traceback.format_exc()}")
            # Clear CUDA cache on error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return None

    def merge_with_transcription(
        self,
        diarization_result: DiarizationResult,
        transcription_result: TranscriptionResult,
    ) -> List[MergedSegment]:
        """
        Merge speaker diarization results with transcription data.

        Args:
            diarization_result: Result from diarize() method
            transcription_result: Result from transcribe_with_timestamps() method

        Returns:
            List of merged segments with speaker and text information
        """
        if not diarization_result or not transcription_result:
            logging.error("Cannot merge: missing diarization or transcription data")
            return []

        diarization_segments = diarization_result.get("segments", [])
        transcription_segments = transcription_result.get("segments", [])

        if not diarization_segments or not transcription_segments:
            logging.warning("Empty diarization or transcription segments")
            return []

        merged_segments = []

        # For each transcription segment, find overlapping speaker segments
        for trans_seg in transcription_segments:
            trans_start = trans_seg.get("start", 0)
            trans_end = trans_seg.get("end", 0)
            trans_text = trans_seg.get("text", "").strip()

            if not trans_text:
                continue

            # Find the speaker who spoke during this transcription segment
            # by finding the diarization segment with the most overlap
            best_speaker = None
            max_overlap = 0

            for diar_seg in diarization_segments:
                diar_start = diar_seg.get("start", 0)
                diar_end = diar_seg.get("end", 0)

                # Calculate overlap
                overlap_start = max(trans_start, diar_start)
                overlap_end = min(trans_end, diar_end)
                overlap = max(0, overlap_end - overlap_start)

                if overlap > max_overlap:
                    max_overlap = overlap
                    best_speaker = diar_seg.get("speaker", "Unknown")

            # If no speaker found, use the most recent speaker or "Unknown"
            if best_speaker is None:
                # Find the closest speaker before this segment
                for diar_seg in reversed(diarization_segments):
                    if diar_seg.get("end", 0) <= trans_start:
                        best_speaker = diar_seg.get("speaker", "Unknown")
                        break
                if best_speaker is None:
                    best_speaker = "Unknown"

            merged_segments.append(
                {
                    "start": trans_start,
                    "end": trans_end,
                    "speaker": best_speaker,
                    "text": trans_text,
                }
            )

        logging.info(f"Merged {len(merged_segments)} segments with speaker information")
        return merged_segments

    def format_diarized_markdown(
        self, merged_segments: List[MergedSegment], title: str = "Meeting Transcript"
    ) -> str:
        """
        Format merged segments as enhanced markdown with speaker labels.

        Example output format:
            **[00:00:00 - Speaker 0]** Hello everyone
            **[00:00:05 - Speaker 1]** Good morning

        Args:
            merged_segments: List of merged segments from merge_with_transcription()
            title: Optional title for the transcript

        Returns:
            Formatted markdown string with speaker labels and timestamps
        """
        if not merged_segments:
            logging.warning("No segments to format")
            return f"# {title}\n\n*No transcription available*\n"

        # Add markdown header
        formatted = f"# {title}\n\n"

        # Format each segment with speaker label and timestamp
        for segment in merged_segments:
            start_time = segment.get("start", 0)
            speaker = segment.get("speaker", "Unknown")
            text = segment.get("text", "").strip()

            if not text:
                continue

            # Format timestamp as HH:MM:SS
            hours = int(start_time // 3600)
            minutes = int((start_time % 3600) // 60)
            seconds = int(start_time % 60)
            timestamp = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

            # Add speaker segment
            formatted += f"**[{timestamp} - {speaker}]** {text}\n\n"

        # Ensure the file ends with a newline
        if not formatted.endswith("\n"):
            formatted += "\n"

        return formatted

    def save_diarized_transcript(
        self,
        merged_segments: List[MergedSegment],
        video_base_name: str,
        title: str = "Meeting Transcript",
    ) -> Optional[Path]:
        """
        Save diarized transcript to a markdown file.

        Args:
            merged_segments: List of merged segments
            video_base_name: Base name of the video file (without extension)
            title: Optional title for the transcript

        Returns:
            Path to the saved transcript file, or None if save failed
        """
        try:
            # Create transcript directory if it doesn't exist
            transcript_dir = Path("transcript")
            transcript_dir.mkdir(parents=True, exist_ok=True)

            # Create transcript file path with diarization suffix
            transcript_path = (
                transcript_dir / f"{video_base_name}_diarized_transcript.md"
            )

            # Format as markdown
            formatted_transcript = self.format_diarized_markdown(merged_segments, title)

            # Write to file
            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write(formatted_transcript)

            logging.info(f"Diarized transcript saved to: {transcript_path}")
            return transcript_path

        except Exception as e:
            logging.error(f"Failed to save diarized transcript: {e}")
            return None

    def get_speaker_summary(self, diarization_result: DiarizationResult) -> Dict:
        """
        Generate a summary of speaker statistics.

        Args:
            diarization_result: Result from diarize() method

        Returns:
            Dictionary with speaker statistics
        """
        if not diarization_result:
            return {}

        segments = diarization_result.get("segments", [])
        if not segments:
            return {}

        # Calculate speaking time per speaker
        speaker_stats = {}
        for segment in segments:
            speaker = segment.get("speaker", "Unknown")
            duration = segment.get("duration", 0)

            if speaker not in speaker_stats:
                speaker_stats[speaker] = {
                    "total_duration": 0,
                    "num_segments": 0,
                }

            speaker_stats[speaker]["total_duration"] += duration
            speaker_stats[speaker]["num_segments"] += 1

        # Sort speakers by total speaking time
        sorted_speakers = sorted(
            speaker_stats.items(),
            key=lambda x: x[1]["total_duration"],
            reverse=True,
        )

        return {
            "speakers": speaker_stats,
            "sorted_speakers": sorted_speakers,
            "total_speakers": len(speaker_stats),
        }
