"""
Voice Activity Detection (VAD) processor for identifying speech segments in audio.

This module provides functionality to:
1. Detect speech segments in audio files using VAD
2. Filter out non-speech segments (silence, background noise)
3. Process only relevant audio segments for transcription
4. Improve transcription accuracy and reduce processing time
"""

import logging
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import webrtcvad
import soundfile as sf
import tempfile
import os
from dataclasses import dataclass

# VAD Preset configurations for different use cases
VAD_PRESETS = {
    "conservative": {
        "aggressiveness": 1,
        "max_silence_duration": 2.0,
        "min_speech_duration": 0.3,
        "description": "Conservative mode - Preserves more audio, less aggressive filtering",
    },
    "moderate": {
        "aggressiveness": 2,
        "max_silence_duration": 1.5,
        "min_speech_duration": 0.5,
        "description": "Moderate mode - Balanced approach for general use",
    },
    "aggressive": {
        "aggressiveness": 3,
        "max_silence_duration": 1.0,
        "min_speech_duration": 0.7,
        "description": "Aggressive mode - Maximum filtering, keeps only clear speech",
    },
}


def get_vad_preset_params(mode: str) -> dict:
    """
    Get VAD parameters for a specific preset mode.

    Args:
        mode: Preset mode name ('conservative', 'moderate', 'aggressive')

    Returns:
        Dictionary containing VAD parameters for the specified mode

    Raises:
        ValueError: If mode is not recognized
    """
    if mode not in VAD_PRESETS:
        available_modes = list(VAD_PRESETS.keys())
        raise ValueError(
            f"Invalid VAD mode '{mode}'. Available modes: {available_modes}"
        )

    return VAD_PRESETS[mode].copy()


@dataclass
class SpeechSegment:
    """Represents a detected speech segment."""

    start_time: float
    end_time: float
    duration: float
    confidence: float = 0.0


class VADProcessor:
    """
    Voice Activity Detection processor using WebRTC VAD and librosa.

    This class provides methods to:
    - Detect speech segments in audio files
    - Filter audio to include only speech segments
    - Generate visualizations of speech activity
    - Process audio for improved transcription
    """

    def __init__(self, aggressiveness: int = 3, sample_rate: int = 16000):
        """
        Initialize VAD processor.

        Args:
            aggressiveness: VAD aggressiveness level (0-3, where 3 is most aggressive)
            sample_rate: Audio sample rate (default: 16000 Hz)
        """
        self.aggressiveness = aggressiveness
        self.sample_rate = sample_rate
        self.vad = webrtcvad.Vad(aggressiveness)

        # Frame duration in milliseconds (WebRTC VAD requires 10, 20, or 30 ms)
        self.frame_duration = 30  # ms

        logging.info(f"Initialized VAD processor with aggressiveness: {aggressiveness}")

    def detect_speech_segments(self, audio_path: Path) -> List[SpeechSegment]:
        """
        Detect speech segments in an audio file.

        Args:
            audio_path: Path to the audio file

        Returns:
            List of SpeechSegment objects containing detected speech segments
        """
        if not audio_path.exists():
            logging.error(f"Audio file not found: {audio_path}")
            return []

        try:
            # Load audio file
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            logging.info(
                f"Loaded audio: {audio_path.name} ({len(audio)/sr:.2f}s, {sr} Hz)"
            )

            # Convert audio to 16-bit PCM format for WebRTC VAD
            audio_int16 = (audio * 32767).astype(np.int16)

            # Calculate frame size in samples
            frame_size = int(self.sample_rate * self.frame_duration / 1000)

            # Split audio into frames
            frames = []
            for i in range(0, len(audio_int16), frame_size):
                frame = audio_int16[i : i + frame_size]
                if len(frame) == frame_size:
                    frames.append(frame)

            logging.info(
                f"Split audio into {len(frames)} frames ({self.frame_duration}ms each)"
            )

            # Detect speech in each frame
            speech_frames = []
            for i, frame in enumerate(frames):
                is_speech = self.vad.is_speech(frame.tobytes(), self.sample_rate)
                if is_speech:
                    speech_frames.append(i)

            if not speech_frames:
                logging.warning("No speech detected in audio")
                return []

            logging.info(
                f"Detected speech in {len(speech_frames)} out of {len(frames)} frames"
            )

            # Group consecutive speech frames into segments
            segments = self._group_frames_into_segments(speech_frames, frame_size)

            logging.info(f"Detected {len(segments)} speech segments")
            for i, segment in enumerate(segments):
                logging.info(
                    f"  Segment {i+1}: {segment.start_time:.2f}s - {segment.end_time:.2f}s ({segment.duration:.2f}s)"
                )

            return segments

        except Exception as e:
            logging.error(f"Error detecting speech segments: {e}")
            return []

    def _group_frames_into_segments(
        self, speech_frames: List[int], frame_size: int
    ) -> List[SpeechSegment]:
        """
        Group consecutive speech frames into speech segments.

        Args:
            speech_frames: List of frame indices containing speech
            frame_size: Size of each frame in samples

        Returns:
            List of SpeechSegment objects
        """
        if not speech_frames:
            return []

        segments = []
        current_segment_start = speech_frames[0]

        for i in range(1, len(speech_frames)):
            # Check if current frame is consecutive to the previous one
            if speech_frames[i] != speech_frames[i - 1] + 1:
                # End current segment and start new one
                current_segment_end = speech_frames[i - 1]
                start_time = current_segment_start * frame_size / self.sample_rate
                end_time = current_segment_end * frame_size / self.sample_rate
                duration = end_time - start_time

                # Only include segments longer than 0.1 seconds
                if duration > 0.1:
                    segments.append(SpeechSegment(start_time, end_time, duration))

                current_segment_start = speech_frames[i]

        # Add the last segment
        current_segment_end = speech_frames[-1]
        start_time = current_segment_start * frame_size / self.sample_rate
        end_time = current_segment_end * frame_size / self.sample_rate
        duration = end_time - start_time

        if duration > 0.1:
            segments.append(SpeechSegment(start_time, end_time, duration))

        return segments

    def filter_audio_to_speech(
        self, audio_path: Path, output_dir: str = "output"
    ) -> Optional[Path]:
        """
        Create a new audio file containing only speech segments.

        Args:
            audio_path: Path to the input audio file
            output_dir: Directory to save the filtered audio

        Returns:
            Path to the filtered audio file, or None if processing failed
        """
        if not audio_path.exists():
            logging.error(f"Audio file not found: {audio_path}")
            return None

        # Detect speech segments
        speech_segments = self.detect_speech_segments(audio_path)
        if not speech_segments:
            logging.warning("No speech segments found, returning original audio")
            return audio_path

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate output filename
        filtered_filename = f"{audio_path.stem}_filtered.wav"
        filtered_path = output_path / filtered_filename

        try:
            # Load original audio
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)

            # Create mask for speech segments
            speech_mask = np.zeros(len(audio), dtype=bool)

            for segment in speech_segments:
                start_sample = int(segment.start_time * sr)
                end_sample = int(segment.end_time * sr)
                speech_mask[start_sample:end_sample] = True

            # Apply mask to keep only speech
            filtered_audio = audio[speech_mask]

            # Save filtered audio
            sf.write(filtered_path, filtered_audio, sr)

            # Calculate statistics
            original_duration = len(audio) / sr
            filtered_duration = len(filtered_audio) / sr
            compression_ratio = filtered_duration / original_duration

            logging.info(f"Filtered audio created: {filtered_filename}")
            logging.info(f"Original duration: {original_duration:.2f}s")
            logging.info(f"Filtered duration: {filtered_duration:.2f}s")
            logging.info(
                f"Compression ratio: {compression_ratio:.2f} ({compression_ratio*100:.1f}% of original)"
            )

            return filtered_path

        except Exception as e:
            logging.error(f"Error creating filtered audio: {e}")
            return None

    def create_spectrogram_visualization(
        self, audio_path: Path, output_dir: str = "output"
    ) -> Optional[Path]:
        """
        Create a spectrogram visualization showing speech activity.

        Args:
            audio_path: Path to the audio file
            output_dir: Directory to save the visualization

        Returns:
            Path to the generated visualization file, or None if failed
        """
        if not audio_path.exists():
            logging.error(f"Audio file not found: {audio_path}")
            return None

        try:
            # Load audio file
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)

            # Detect speech segments
            speech_segments = self.detect_speech_segments(audio_path)

            # Create output directory
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Generate output filename
            viz_filename = f"{audio_path.stem}_speech_activity.png"
            viz_path = output_path / viz_filename

            # Create spectrogram
            plt.figure(figsize=(12, 8))

            # Plot spectrogram
            D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
            librosa.display.specshow(D, sr=sr, x_axis="time", y_axis="log")
            plt.colorbar(format="%+2.0f dB")
            plt.title("Speech Activity Detection")
            plt.xlabel("Time (s)")
            plt.ylabel("Frequency (Hz)")

            # Add speech segment overlays
            if speech_segments:
                for segment in speech_segments:
                    plt.axvspan(
                        segment.start_time,
                        segment.end_time,
                        alpha=0.3,
                        color="green",
                        label="Speech" if segment == speech_segments[0] else "",
                    )

                # Add legend
                plt.legend()

            # Save the plot
            plt.tight_layout()
            plt.savefig(viz_path, dpi=150, bbox_inches="tight")
            plt.close()

            logging.info(f"Speech activity visualization created: {viz_filename}")
            return viz_path

        except Exception as e:
            logging.error(f"Error creating visualization: {e}")
            return None

    def get_speech_statistics(self, audio_path: Path) -> Dict[str, float]:
        """
        Get statistics about speech activity in the audio file.

        Args:
            audio_path: Path to the audio file

        Returns:
            Dictionary containing speech statistics
        """
        if not audio_path.exists():
            logging.error(f"Audio file not found: {audio_path}")
            return {}

        try:
            # Load audio file
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            total_duration = len(audio) / sr

            # Detect speech segments
            speech_segments = self.detect_speech_segments(audio_path)

            if not speech_segments:
                return {
                    "total_duration": total_duration,
                    "speech_duration": 0.0,
                    "speech_ratio": 0.0,
                    "num_segments": 0,
                    "avg_segment_duration": 0.0,
                    "longest_segment": 0.0,
                }

            # Calculate statistics
            speech_duration = sum(segment.duration for segment in speech_segments)
            speech_ratio = speech_duration / total_duration
            num_segments = len(speech_segments)
            avg_segment_duration = speech_duration / num_segments
            longest_segment = max(segment.duration for segment in speech_segments)

            return {
                "total_duration": total_duration,
                "speech_duration": speech_duration,
                "speech_ratio": speech_ratio,
                "num_segments": num_segments,
                "avg_segment_duration": avg_segment_duration,
                "longest_segment": longest_segment,
            }

        except Exception as e:
            logging.error(f"Error calculating speech statistics: {e}")
            return {}

    def get_vad_segments(
        self,
        audio_path: str,
        aggressiveness: int = 3,
        frame_duration_ms: int = 30,
    ) -> List[Tuple[float, float]]:
        """
        Get VAD segments from audio file.

        Args:
            audio_path: Path to audio file
            aggressiveness: VAD aggressiveness level (0-3)
            frame_duration_ms: Frame duration in milliseconds

        Returns:
            List of (start_time, end_time) tuples for speech segments
        """
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=16000)

            # Initialize VAD
            vad = webrtcvad.Vad(aggressiveness)

            # Calculate frame length
            frame_length = int(sr * frame_duration_ms / 1000)

            # Get speech segments
            segments = []
            start_time = None

            for i in range(0, len(audio), frame_length):
                frame = audio[i : i + frame_length]
                if len(frame) < frame_length:
                    # Pad the last frame
                    frame = np.pad(
                        frame, (0, frame_length - len(frame)), mode="constant"
                    )

                # Check if frame is speech
                is_speech = vad.is_speech(frame.tobytes(), sr)

                current_time = i / sr

                if is_speech and start_time is None:
                    start_time = current_time
                elif not is_speech and start_time is not None:
                    segments.append((start_time, current_time))
                    start_time = None

            # Add final segment if audio ends with speech
            if start_time is not None:
                segments.append((start_time, len(audio) / sr))

            return segments

        except Exception as e:
            logging.error(f"Error getting VAD segments: {e}")
            return []


def moderate_silence_cancellation(
    audio_path: str,
    aggressiveness: int = 2,
    max_silence_duration: float = 1.5,
    min_speech_duration: float = 0.5,
    output_dir: str = "output",
    mode: str = None,
) -> Tuple[str, float]:
    """
    Apply moderate silence cancellation to audio file.

    This function implements a balanced approach that removes:
    - Long silences (> max_silence_duration) between speech segments
    - Very short speech segments (< min_speech_duration)
    - Preserves natural speech patterns and moderate pauses

    Args:
        audio_path: Path to the input audio file
        aggressiveness: VAD aggressiveness level (0-3) - ignored if mode is provided
        max_silence_duration: Maximum silence duration to preserve (seconds) - ignored if mode is provided
        min_speech_duration: Minimum speech segment duration to keep (seconds) - ignored if mode is provided
        output_dir: Directory to save output files
        mode: Preset mode name ('conservative', 'moderate', 'aggressive') - overrides other parameters

    Returns:
        Tuple of (processed_audio_path, total_speech_duration)
    """
    try:
        # Use preset parameters if mode is specified
        if mode:
            preset_params = get_vad_preset_params(mode)
            aggressiveness = preset_params["aggressiveness"]
            max_silence_duration = preset_params["max_silence_duration"]
            min_speech_duration = preset_params["min_speech_duration"]
            logging.info(f"Using VAD preset: {mode} - {preset_params['description']}")

        # Initialize VAD processor
        vad_processor = VADProcessor(aggressiveness=aggressiveness)

        logging.info(
            f"VAD parameters: aggressiveness={aggressiveness}, max_silence={max_silence_duration}s, min_speech={min_speech_duration}s"
        )

        # Detect speech segments
        audio_path_obj = Path(audio_path)
        speech_segments = vad_processor.detect_speech_segments(audio_path_obj)

        if not speech_segments:
            logging.warning("No speech segments found, returning original audio")
            return str(audio_path), 0.0

        # Load original audio
        y, sr = librosa.load(audio_path, sr=None)
        original_duration = len(y) / sr

        # Process segments with moderate cancellation
        final_segments = []

        for i, segment in enumerate(speech_segments):
            # Filter out very short speech segments
            if segment.duration < min_speech_duration:
                logging.debug(f"Removing short speech segment: {segment.duration:.2f}s")
                continue

            # Add segment to final list
            final_segments.append((segment.start_time, segment.end_time))

            # Check if there's a following segment
            if i < len(speech_segments) - 1:
                next_segment = speech_segments[i + 1]
                silence_duration = next_segment.start_time - segment.end_time

                # Remove long silences between segments
                if silence_duration > max_silence_duration:
                    logging.debug(f"Removing long silence: {silence_duration:.2f}s")
                    # Skip the silence, don't add anything
                    continue
                else:
                    # Keep moderate silences
                    logging.debug(f"Keeping moderate silence: {silence_duration:.2f}s")
                    final_segments.append((segment.end_time, next_segment.start_time))

        # Calculate total speech duration
        total_speech_duration = sum(end - start for start, end in final_segments)
        compression_ratio = total_speech_duration / original_duration

        logging.info(f"Original duration: {original_duration:.2f}s")
        logging.info(f"Processed duration: {total_speech_duration:.2f}s")
        logging.info(
            f"Compression ratio: {compression_ratio:.2f} ({compression_ratio*100:.1f}% of original)"
        )

        # Create processed audio by concatenating segments
        processed_audio = np.array([])
        for start, end in final_segments:
            start_sample = int(start * sr)
            end_sample = int(end * sr)
            segment_audio = y[start_sample:end_sample]
            processed_audio = np.concatenate([processed_audio, segment_audio])

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate output filename
        processed_filename = f"{audio_path_obj.stem}_moderate_processed.wav"
        output_path_obj = output_path / processed_filename

        # Save processed audio
        sf.write(output_path_obj, processed_audio, sr)
        logging.info(f"Moderately processed audio saved to: {output_path_obj}")

        # Create visualization
        try:
            plt.figure(figsize=(12, 8))

            # Plot original waveform
            plt.subplot(2, 1, 1)
            librosa.display.waveshow(y, sr=sr, alpha=0.7)
            plt.title(f"Original Audio - Duration: {original_duration:.2f}s")
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude")

            # Plot processed waveform
            plt.subplot(2, 1, 2)
            librosa.display.waveshow(processed_audio, sr=sr, alpha=0.7, color="orange")
            plt.title(
                f"Moderately Processed Audio - Duration: {total_speech_duration:.2f}s ({compression_ratio:.1f}% compression)"
            )
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude")

            plt.tight_layout()

            # Save visualization
            viz_filename = f"{audio_path_obj.stem}_moderate_processing_viz.png"
            viz_path = output_path / viz_filename
            plt.savefig(viz_path, dpi=150, bbox_inches="tight")
            plt.close()

            logging.info(f"Moderate processing visualization saved to: {viz_path}")

        except Exception as e:
            logging.warning(f"Could not create visualization: {e}")

        return str(output_path_obj), total_speech_duration

    except Exception as e:
        logging.error(f"Error in moderate silence cancellation: {e}")
        return str(audio_path), 0.0


def process_audio_with_vad(
    audio_path: str,
    aggressiveness: int = 2,
    max_silence_duration: float = 1.5,
    min_speech_duration: float = 0.5,
    create_visualization: bool = True,
    output_dir: str = "output",
    mode: str = None,
) -> Tuple[str, float]:
    """
    Process audio file with moderate silence cancellation and return the processed audio path and duration.

    Args:
        audio_path: Path to the input audio file
        aggressiveness: VAD aggressiveness level (0-3) - ignored if mode is provided
        max_silence_duration: Maximum silence duration to preserve (seconds) - ignored if mode is provided
        min_speech_duration: Minimum speech segment duration to keep (seconds) - ignored if mode is provided
        create_visualization: Whether to create a visualization plot
        output_dir: Directory to save output files
        mode: Preset mode name ('conservative', 'moderate', 'aggressive') - overrides other parameters

    Returns:
        Tuple of (processed_audio_path, total_speech_duration)
    """
    return moderate_silence_cancellation(
        audio_path=audio_path,
        aggressiveness=aggressiveness,
        max_silence_duration=max_silence_duration,
        min_speech_duration=min_speech_duration,
        output_dir=output_dir,
        mode=mode,
    )


# =============================================================================
# PARAMETER TUNING GUIDE
# =============================================================================
"""
VOICE ACTIVITY DETECTION (VAD) PARAMETER TUNING GUIDE
======================================================

This guide helps you understand and configure VAD parameters for optimal performance.

## VAD AGGRESSIVENESS LEVELS (0-3)
----------------------------------
The aggressiveness parameter controls how sensitive the VAD is to detecting speech:

### Level 0 (Conservative)
- Lowest sensitivity to speech
- Best for: High-quality audio with clear speech and minimal background noise
- Use case: Studio recordings, clean audio with little to no background noise
- May miss: Soft speech, distant speakers, speech with heavy accents

### Level 1 (Moderate-Conservative)
- Low sensitivity to speech
- Best for: Good quality audio with some background noise
- Use case: Professional meetings, clear phone calls
- Balance: Fewer false positives while capturing most speech

### Level 2 (Moderate) ⭐ RECOMMENDED
- Medium sensitivity to speech
- Best for: Most real-world scenarios with moderate background noise
- Use case: Standard meetings, video conferences, typical audio recordings
- Balance: Good trade-off between false positives and missed speech
- This is the default and recommended setting for most applications

### Level 3 (Aggressive)
- Highest sensitivity to speech
- Best for: Noisy environments with significant background noise
- Use case: Crowded rooms, outdoor recordings, low-quality audio
- Warning: May incorrectly classify some non-speech sounds as speech

## SILENCE CANCELLATION PARAMETERS
----------------------------------

### max_silence_duration
Maximum duration of silence to preserve between speech segments (in seconds).

- Small values (0.5-1.0s): Aggressive silence removal, keeps only continuous speech
- Medium values (1.5-2.0s): Moderate silence removal, preserves natural pauses ⭐
- Large values (2.5-3.0s): Conservative silence removal, keeps most pauses

### min_speech_duration
Minimum duration of a speech segment to keep (in seconds).

- Small values (0.1-0.3s): Keeps very short utterances, coughs, breaths
- Medium values (0.5-0.7s): Filters out very short sounds, keeps normal speech ⭐
- Large values (1.0-2.0s): Only keeps longer speech segments

## PRESET MODES
--------------

### Conservative Mode
- Aggressiveness: 1
- Max Silence: 2.0s
- Min Speech: 0.3s
- Use when: You want to preserve as much audio as possible
- Best for: High-quality recordings where every word matters

### Moderate Mode (Default) ⭐
- Aggressiveness: 2
- Max Silence: 1.5s
- Min Speech: 0.5s
- Use when: You want a balanced approach for general use
- Best for: Most meeting recordings, standard audio quality

### Aggressive Mode
- Aggressiveness: 3
- Max Silence: 1.0s
- Min Speech: 0.7s
- Use when: You want maximum filtering and only clear speech
- Best for: Noisy environments, when you need only the clearest speech

## TROUBLESHOOTING
-----------------

### Problem: Too much background noise in processed audio
Solutions:
- Increase aggressiveness level (try 3)
- Decrease max_silence_duration (try 1.0s)
- Increase min_speech_duration (try 0.7s)
- Use 'aggressive' preset mode

### Problem: Missing speech segments or cutting off words
Solutions:
- Decrease aggressiveness level (try 1 or 0)
- Increase max_silence_duration (try 2.0s)
- Decrease min_speech_duration (try 0.3s)
- Use 'conservative' preset mode

### Problem: Audio sounds unnatural or choppy
Solutions:
- Increase max_silence_duration to preserve natural pauses
- Use 'moderate' or 'conservative' preset
- Check if aggressiveness is too high

### Problem: Very short processing time or no audio output
Solutions:
- Check if audio file is valid and has speech
- Decrease aggressiveness level
- Decrease min_speech_duration
- Verify audio file format (should be WAV, MP3, M4A, etc.)

### Problem: Processing is too slow
Solutions:
- Ensure audio is not excessively long (consider splitting)
- Check system resources (CPU, RAM)
- Verify dependencies are properly installed

## PERFORMANCE OPTIMIZATION
--------------------------

1. **Audio Quality**: Higher quality audio (16kHz+, 16-bit) works best
2. **File Format**: WAV files process faster than compressed formats
3. **Audio Length**: Consider splitting very long audio files (>1 hour)
4. **Sample Rate**: 16000 Hz is optimal for WebRTC VAD

## EXAMPLE USAGE
---------------

```python
from vad_processor import process_audio_with_vad

# Using preset mode (recommended)
processed_audio, duration = process_audio_with_vad(
    audio_path="meeting.wav",
    mode="moderate"  # or "conservative", "aggressive"
)

# Using custom parameters
processed_audio, duration = process_audio_with_vad(
    audio_path="meeting.wav",
    aggressiveness=2,
    max_silence_duration=1.5,
    min_speech_duration=0.5
)
```

## ADVANCED CONFIGURATION
------------------------

For fine-tuning, you can access presets directly:

```python
from vad_processor import get_vad_preset_params

params = get_vad_preset_params('moderate')
# Returns: {'aggressiveness': 2, 'max_silence_duration': 1.5, ...}
```

Then modify individual parameters as needed for your specific use case.
"""


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(
        description="Process audio with Voice Activity Detection"
    )
    parser.add_argument("audio_file", help="Path to the audio file to process")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument(
        "--aggressiveness",
        type=int,
        default=3,
        choices=[0, 1, 2, 3],
        help="VAD aggressiveness level (0-3)",
    )

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Process audio
    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}")
        exit(1)

    processed_audio, total_duration = process_audio_with_vad(
        audio_path=str(audio_path),
        aggressiveness=args.aggressiveness,
        max_silence_duration=1.5,
        min_speech_duration=0.5,
        output_dir=args.output_dir,
    )

    if processed_audio:
        print(f"Processed audio saved to: {processed_audio}")
        print(f"Total speech duration: {total_duration:.2f}s")
