#!/usr/bin/env python3
"""
Meeting Summary Generator

A Python application that processes MP4 video files to generate meeting summaries.
Workflow: MP4 → Audio Extraction → Transcription → Summary Generation → Markdown Output
"""

import sys
import logging
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils import (
    setup_logging,
    get_most_recent_video,
    ensure_output_dir,
    construct_output_path,
    check_dependencies,
)
from video_processor import extract_audio_from_video, cleanup_audio_file
from transcriber import Transcriber
from summarizer import Summarizer


def main():
    """Main application entry point."""
    # Set up logging
    setup_logging()
    logging.info("=" * 60)
    logging.info("Meeting Summary Generator - Starting")
    logging.info("=" * 60)

    # Configuration
    VIDEO_DIR = "/mnt/c/Users/danis/Videos"  # Windows Videos folder in WSL
    OUTPUT_DIR = "output"
    WHISPER_MODEL = "medium"
    OLLAMA_MODEL = "qwen2.5:7b"
    CLEANUP_AUDIO = True  # Set to False to keep intermediate audio files

    # Step 1: Check dependencies
    logging.info("\n[Step 1/6] Checking dependencies...")
    if not check_dependencies():
        logging.error("Dependency check failed. Please install required dependencies.")
        logging.info("Run: pip install -r requirements.txt")
        logging.info("Also ensure ffmpeg is installed and Ollama is running")
        return 1

    # Step 2: Get most recent video file
    logging.info("\n[Step 2/6] Finding most recent video file...")
    video_path = get_most_recent_video(VIDEO_DIR)
    if not video_path:
        logging.error(f"No video files found in {VIDEO_DIR} directory")
        logging.info(
            f"Please place your video meeting recording in the {VIDEO_DIR} folder"
        )
        return 1

    logging.info(f"Processing: {video_path.name}")

    # Step 3: Ensure output directory exists
    logging.info("\n[Step 3/6] Preparing output directory...")
    if not ensure_output_dir(OUTPUT_DIR):
        logging.error("Failed to create output directory")
        return 1

    # Step 4: Extract audio from video
    logging.info("\n[Step 4/6] Extracting audio from video...")
    audio_path = extract_audio_from_video(video_path, OUTPUT_DIR)
    if not audio_path:
        logging.error("Failed to extract audio from video")
        return 1

    # Step 5: Transcribe audio
    logging.info("\n[Step 5/6] Transcribing audio...")
    transcriber = Transcriber(model_size=WHISPER_MODEL)

    if not transcriber.load_model():
        logging.error("Failed to load Whisper model")
        if CLEANUP_AUDIO:
            cleanup_audio_file(audio_path)
        return 1

    # Get video base name for transcript file
    video_base_name = video_path.stem

    # Transcribe audio and save to transcript folder
    transcription = transcriber.transcribe_audio(audio_path, video_base_name)
    if not transcription:
        logging.error("Failed to transcribe audio")
        if CLEANUP_AUDIO:
            cleanup_audio_file(audio_path)
        return 1

    logging.info("Transcription preview (first 200 chars):")
    logging.info(f"  {transcription[:200]}...")

    # Clean up audio file if requested
    if CLEANUP_AUDIO:
        cleanup_audio_file(audio_path)

    # Step 6: Generate summary
    logging.info("\n[Step 6/6] Generating meeting summary...")
    summarizer = Summarizer(model_name=OLLAMA_MODEL)

    # Check if model is available, pull if not
    if not summarizer.check_model_available():
        logging.info(f"Model '{OLLAMA_MODEL}' not found. Attempting to pull...")
        if not summarizer.pull_model():
            logging.error(f"Failed to pull model '{OLLAMA_MODEL}'")
            logging.info(
                f"Please ensure Ollama is running and the model name is correct"
            )
            return 1

    # Generate summary from transcript file
    meeting_title = video_path.stem.replace("_", " ").replace("-", " ")
    transcript_path = Path("transcript") / f"{video_base_name}_transcript.md"

    logging.info(f"Reading transcript from: {transcript_path}")
    summary = summarizer.generate_summary(transcript_path, meeting_title)
    if not summary:
        logging.error("Failed to generate summary")
        return 1

    # Save summary to file
    summary_path = construct_output_path(video_path, OUTPUT_DIR, ".md")
    if not summarizer.save_summary(summary, summary_path):
        logging.error("Failed to save summary")
        return 1

    # Success!
    logging.info("\n" + "=" * 60)
    logging.info("SUCCESS! Meeting summary generated")
    logging.info("=" * 60)
    logging.info(f"Input video:    {video_path.name}")
    logging.info(f"Transcript:     {transcript_path.name}")
    logging.info(f"Output file:    {summary_path.name}")
    logging.info(f"Summary location: {summary_path}")
    logging.info(f"Transcript location: {transcript_path}")
    logging.info("=" * 60)

    # Show summary preview
    logging.info("\nSummary preview (first 500 chars):")
    logging.info("-" * 60)
    logging.info(summary[:500])
    if len(summary) > 500:
        logging.info("...")
    logging.info("-" * 60)

    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logging.warning("\n\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        logging.error(f"\n\nUnexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
