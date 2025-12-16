#!/usr/bin/env python3
"""
Meeting Summary Generator

A Python application that processes MP4 video files to generate meeting summaries.
Workflow: MP4 → Audio Extraction → Transcription → Summary Generation → Markdown Output
"""

import sys
import logging
import time
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

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
from transcriber import Transcriber, WhisperHinglishTranscriber
from summarizer import Summarizer
from diarizer import Diarizer

# Initialize Rich console and timing tracker
console = Console()
timing_data = {
    "video_extraction": 0,
    "stt_transcription": 0,
    "speaker_diarization": 0,
    "llm_summarization": 0,
    "total": 0,
}


def get_video_from_input_folder():
    """Check input folder for video files and return the path if exactly one exists"""
    input_dir = Path("input")

    if not input_dir.exists():
        input_dir.mkdir(exist_ok=True)
        return None

    # Supported video extensions
    video_extensions = {
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".flv",
        ".wmv",
        ".webm",
        ".m4v",
        ".mpg",
        ".mpeg",
    }

    # Find all video files in input folder
    video_files = [
        f
        for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in video_extensions
    ]

    if len(video_files) == 0:
        console.print(
            "[yellow]📂 Input folder is empty. Will prompt for video file.[/yellow]\n"
        )
        return None
    elif len(video_files) == 1:
        video_path = video_files[0]
        console.print(
            f"✅ [green]Found video in input folder:[/green] {video_path.name}"
        )
        return video_path
    else:
        console.print(
            "[red]❌ Error: Multiple video files found in input folder![/red]"
        )
        console.print("[red]Please keep only one video file in the input folder:[/red]")
        for vf in video_files:
            console.print(f"   • {vf.name}")
        console.print(
            f"\n[yellow]📂 Input folder location:[/yellow] {input_dir.absolute()}"
        )
        return "multiple_files_error"


def play_completion_sound():
    """Play a sound notification when processing is complete"""
    try:
        import os

        if os.name == "nt":  # Windows
            import winsound

            winsound.MessageBeep(winsound.MB_OK)
        else:  # Linux/Mac
            # Bell character - most terminals will make a beep sound
            print("\a")

        console.print("🔔 [dim]Completion sound played[/dim]")
    except Exception as e:
        # If sound fails, just continue silently
        console.print(f"[dim]Note: Could not play sound ({e})[/dim]")


def format_duration(seconds):
    """Format duration in seconds to human-readable format"""
    if seconds < 60:
        return f"{seconds:.2f}s"
    else:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}:{remaining_seconds:05.2f}"


def display_summary(timing_data, video_path, transcript_path, summary_path):
    """Display beautiful summary with Rich"""

    console.print("\n")
    console.print(
        Panel.fit(
            "🎬 [bold cyan]Meeting Processing Complete![/bold cyan] 🎬", style="cyan"
        )
    )
    console.print("\n")

    # Timing Table
    table = Table(title="⏱️ Processing Times", box=box.ROUNDED)
    table.add_column("Step", style="cyan", no_wrap=True)
    table.add_column("Duration", style="magenta", justify="right")
    table.add_column("Percentage", style="green", justify="right")

    total_time = timing_data["total"]

    for step, duration in timing_data.items():
        if step != "total" and duration > 0:
            percentage = (duration / total_time) * 100
            table.add_row(
                step.replace("_", " ").title(),
                f"{format_duration(duration)}",
                f"{percentage:.1f}%",
            )

    table.add_row("", "", "")
    table.add_row(
        "[bold]Total Time[/bold]", f"[bold]{format_duration(total_time)}[/bold]", "100%"
    )

    console.print(table)
    console.print("\n")

    # File Locations
    file_table = Table(title="📁 Output Files", box=box.SIMPLE)
    file_table.add_column("Type", style="blue")
    file_table.add_column("Location", style="white")

    file_table.add_row("Video", str(video_path))
    file_table.add_row("Transcript", f"[green]{transcript_path}[/green]")
    file_table.add_row("Summary", f"[green]{summary_path}[/green]")

    console.print(file_table)
    console.print("\n")

    # Performance Insights
    insights = []
    if timing_data["stt_transcription"] > 0:
        insights.append(
            f"• STT Model: {format_duration(timing_data['stt_transcription'])}"
        )
    if timing_data["llm_summarization"] > 0:
        insights.append(
            f"• LLM Summary: {format_duration(timing_data['llm_summarization'])}"
        )
    if timing_data["video_extraction"] > 0:
        insights.append(
            f"• Audio Extraction: {format_duration(timing_data['video_extraction'])}"
        )

    if insights:
        console.print(
            Panel(
                "\n".join(insights),
                title="📊 Performance Metrics",
                border_style="yellow",
            )
        )

    console.print(
        "\n✅ [bold green]All done! Check your transcript and summary files.[/bold green]\n"
    )

    # Play completion sound
    play_completion_sound()


def main():
    """Main application entry point."""
    # Set up logging
    setup_logging()
    logging.info("=" * 60)
    logging.info("Meeting Summary Generator - Starting")
    logging.info("=" * 60)

    # Ensure input directory exists
    input_dir = Path("input")
    input_dir.mkdir(exist_ok=True)
    console.print(f"📁 [dim]Input directory: {input_dir.absolute()}[/dim]")

    # Configuration
    VIDEO_DIR = "/mnt/c/Users/danis/Videos"  # Windows Videos folder in WSL
    OUTPUT_DIR = "output"
    OLLAMA_MODEL = "qwen2.5:7b"
    CLEANUP_AUDIO = True  # Set to False to keep intermediate audio files
    ENABLE_DIARIZATION = False  # Set to True to enable speaker diarization

    # Model selection
    print("\n" + "=" * 60)
    print("Choose STT Model:")
    print("=" * 60)
    print("1. Whisper (OpenAI) - Multi-language, general purpose")
    print(
        "2. Whisper-Hinglish (Oriserve) - Specialized for Hinglish (Latin script output)"
    )
    print("=" * 60)

    choice = input("Enter choice (1 or 2): ").strip()

    if choice == "1":
        # Existing Whisper model
        model_size = input(
            "Enter Whisper model size (tiny, base, small, medium, large) [medium]: "
        ).strip()
        if not model_size:
            model_size = "medium"
        transcriber = Transcriber(model_size=model_size)
        model_name = f"Whisper ({model_size})"
    elif choice == "2":
        # New Hinglish model
        transcriber = WhisperHinglishTranscriber()
        model_name = "Whisper-Hinglish"
    else:
        print("Invalid choice. Defaulting to Whisper (medium)")
        transcriber = Transcriber(model_size="medium")
        model_name = "Whisper (medium)"

    console.print(f"🤖 [blue]Selected Model:[/blue] {model_name}")

    # Ask user if they want to enable speaker diarization
    print("\n" + "=" * 60)
    print("Speaker Diarization:")
    print("=" * 60)
    print("Enable speaker diarization to identify who spoke when?")
    print("Note: This requires additional processing time and GPU memory")
    print("=" * 60)
    diarize_choice = input("Enable speaker diarization? (y/n) [n]: ").strip().lower()

    logging.info(f"User input for diarization: '{diarize_choice}'")

    if diarize_choice in ["y", "yes"]:
        ENABLE_DIARIZATION = True
        console.print("✅ [green]Speaker diarization enabled[/green]")
        logging.info("User enabled speaker diarization")
    else:
        ENABLE_DIARIZATION = False
        console.print("[dim]Speaker diarization disabled[/dim]")
        logging.info("User disabled speaker diarization")

    print("=" * 60)

    # Step 1: Check dependencies
    logging.info("\n[Step 1/6] Checking dependencies...")
    console.print("🔍 [blue]Checking dependencies...[/blue]")
    if not check_dependencies():
        console.print("❌ [red]Dependency check failed![/red]")
        logging.error("Dependency check failed. Please install required dependencies.")
        logging.info("Run: pip install -r requirements.txt")
        logging.info("Also ensure ffmpeg is installed and Ollama is running")
        return 1
    console.print("✅ [green]All dependencies satisfied![/green]")

    # Step 2: Get video file from input folder or prompt user
    logging.info("\n[Step 2/6] Finding video file...")
    console.print("🎥 [blue]Finding video file...[/blue]")

    # Check for video in input folder
    video_path = get_video_from_input_folder()

    if video_path == "multiple_files_error":
        console.print("\n[red]Exiting due to multiple video files.[/red]")
        return 1
    elif video_path:
        # Use the video from input folder
        console.print(f"🎬 [cyan]Processing:[/cyan] {video_path.name}")
        logging.info(f"Processing: {video_path.name}")
    else:
        # Fall back to Windows Videos folder - get most recent video
        console.print(
            "[yellow]📂 No video found in input folder. Checking Windows Videos folder...[/yellow]"
        )
        logging.info(f"Checking Windows Videos folder: {VIDEO_DIR}")
        video_path = get_most_recent_video(VIDEO_DIR)

        if not video_path:
            console.print(
                f"❌ [red]No video files found in {VIDEO_DIR} directory![/red]"
            )
            logging.error(f"No video files found in {VIDEO_DIR} directory")
            logging.info(
                f"Please place your video meeting recording in the {VIDEO_DIR} folder"
            )
            return 1

        console.print(
            f"✅ [green]Video loaded successfully from Windows Videos folder![/green] {video_path.name}"
        )
        logging.info(f"Processing: {video_path.name}")

    # Check file size and warn if large
    video_size_mb = video_path.stat().st_size / (1024 * 1024)
    if video_size_mb > 100:
        console.print(
            f"⚠️ [yellow]Warning: Large video file detected ({video_size_mb:.1f} MB)[/yellow]"
        )
        console.print(f"   [dim]Processing may take longer than usual.[/dim]")

    # Step 3: Ensure output directory exists
    logging.info("\n[Step 3/6] Preparing output directory...")
    if not ensure_output_dir(OUTPUT_DIR):
        logging.error("Failed to create output directory")
        return 1

    # Step 4: Extract audio from video
    logging.info("\n[Step 4/6] Extracting audio from video...")
    console.print("🎵 [blue]Extracting audio from video...[/blue]")
    start_time = time.time()
    audio_path = extract_audio_from_video(video_path, OUTPUT_DIR)
    timing_data["video_extraction"] = time.time() - start_time
    if not audio_path:
        logging.error("Failed to extract audio from video")
        return 1
    console.print(
        f"✅ [green]Audio extracted in {timing_data['video_extraction']:.2f}s[/green]"
    )

    # Step 5: Transcribe audio
    logging.info("\n[Step 5/6] Transcribing audio...")

    # Load the selected model with timeout handling
    console.print(f"📦 [blue]Loading {type(transcriber).__name__}...[/blue]")
    logging.info(f"Loading {type(transcriber).__name__}...")
    if not transcriber.load_model():
        console.print(f"❌ [red]Failed to load {type(transcriber).__name__}![/red]")
        logging.error(f"Failed to load {type(transcriber).__name__}")
        if CLEANUP_AUDIO:
            cleanup_audio_file(audio_path)
        return 1
    console.print(
        f"✅ [green]{type(transcriber).__name__} loaded successfully![/green]"
    )

    # Get video base name for transcript file
    video_base_name = video_path.stem

    # Transcribe audio and save to transcript folder
    with console.status("[bold green]Transcribing audio...[/bold green]") as status:
        start_time = time.time()
        transcription = transcriber.transcribe_audio(audio_path, video_base_name)
        stt_time = time.time() - start_time
        timing_data["stt_transcription"] = stt_time

    if not transcription:
        logging.error("Failed to transcribe audio")
        if CLEANUP_AUDIO:
            cleanup_audio_file(audio_path)
        return 1

    console.print(f"✅ [green]Transcription completed in {stt_time:.2f}s[/green]")

    logging.info("Transcription preview (first 200 chars):")
    logging.info(f"  {transcription[:200]}...")

    # Step 6: Optional Speaker Diarization
    transcript_path = None
    if ENABLE_DIARIZATION:
        logging.info("\n[Step 6/7] Performing speaker diarization...")
        console.print("👥 [blue]Performing speaker diarization...[/blue]")
        logging.info(f"ENABLE_DIARIZATION flag is True, proceeding with diarization")

        # Initialize diarizer
        logging.info("Initializing Diarizer...")

        # Check for Hugging Face token in environment
        import os
        from dotenv import load_dotenv

        load_dotenv()

        hf_token = os.environ.get("HF_TOKEN")
        logging.info(f"HF_TOKEN from environment: {'Set' if hf_token else 'Not set'}")

        if hf_token:
            diarizer = Diarizer(token=hf_token)
            logging.info("Diarizer initialized with HF_TOKEN from environment")
        else:
            diarizer = Diarizer()
            logging.warning(
                "Diarizer initialized without HF_TOKEN - this may cause authentication errors"
            )
            logging.warning(
                "Set HF_TOKEN environment variable to avoid authentication issues"
            )

        # Check if diarizer was properly initialized (dependencies available)
        if diarizer.pipeline is None and not hasattr(diarizer, "device"):
            # This means dependencies are missing
            console.print(
                "\n❌ [red]Speaker diarization dependencies are not installed![/red]"
            )
            console.print(
                "[yellow]To enable speaker diarization, install the required dependencies:[/yellow]"
            )
            console.print("   pip install torch>=2.0.0")
            console.print(
                "   pip install 'pyannote.audio @ git+https://github.com/pyannote/pyannote-audio.git'"
            )
            console.print("   pip install pyannote.core librosa soundfile")
            console.print(
                "\n[yellow]You also need to accept the model terms on Hugging Face:[/yellow]"
            )
            console.print(
                "   1. Visit https://huggingface.co/pyannote/speaker-diarization-3.1"
            )
            console.print("   2. Click 'Access repository' and accept the terms")
            console.print("   3. Wait 5-10 minutes for permission to propagate")
            console.print(
                "   4. Set your token in .env file: HF_TOKEN='your_token_here'"
            )
            console.print(
                "\n💡 [cyan]Tip:[/cyan] If you don't want to authenticate, the app will"
            )
            console.print(
                "   automatically fall back to pyannote/speaker-diarization-2.0"
            )
            console.print(
                "\n[yellow]Continuing without speaker diarization...[/yellow]\n"
            )
            transcript_path = Path("transcript") / f"{video_base_name}_transcript.md"
            logging.warning(
                "Diarization dependencies missing, continuing without diarization"
            )
            return

        logging.info("Diarizer initialized")

        # Load diarization model
        logging.info("Loading diarization model...")
        with console.status(
            "[bold green]Loading diarization model...[/bold green]"
        ) as status:
            model_loaded = diarizer.load_model()
            logging.info(f"Model loading result: {model_loaded}")

            if not model_loaded:
                console.print("\n[yellow]⚠️  Diarization model failed to load.[/yellow]")
                console.print("[yellow]Possible reasons:[/yellow]")
                console.print(
                    "   • Model authentication failed (check HF_TOKEN in .env)"
                )
                console.print("   • Haven't accepted model terms on Hugging Face")
                console.print("   • Insufficient GPU/CPU memory")
                console.print("   • Network connectivity issues")
                console.print(
                    "\n💡 [cyan]Tip:[/cyan] Visit https://huggingface.co/pyannote/speaker-diarization-3.1"
                )
                console.print("   Click 'Access repository' and accept the terms")
                console.print(
                    "\n[yellow]Continuing without speaker diarization...[/yellow]\n"
                )
                logging.warning(
                    "Diarization model failed to load. Continuing without diarization."
                )
                transcript_path = (
                    Path("transcript") / f"{video_base_name}_transcript.md"
                )
                logging.info(f"Setting transcript_path to: {transcript_path}")
            else:
                logging.info("Model loaded successfully, proceeding with diarization")
                # Perform diarization
                logging.info("Starting diarization process...")
                with console.status(
                    "[bold green]Identifying speakers...[/bold green]"
                ) as status:
                    start_time = time.time()
                    diarization_result = diarizer.diarize(audio_path)
                    diarization_time = time.time() - start_time
                    timing_data["speaker_diarization"] = diarization_time
                    logging.info(f"Diarization took {diarization_time:.2f} seconds")

                logging.info(f"Diarization result: {diarization_result is not None}")

                if diarization_result:
                    console.print(
                        f"✅ [green]Diarization completed in {diarization_time:.2f}s[/green]"
                    )
                    logging.info(
                        f"Identified {diarization_result['num_speakers']} speakers"
                    )

                    # Get transcription with timestamps for merging
                    logging.info("Getting transcription with timestamps...")
                    with console.status(
                        "[bold green]Getting transcription with timestamps...[/bold green]"
                    ) as status:
                        transcription_result = transcriber.transcribe_with_timestamps(
                            audio_path
                        )
                        logging.info(
                            f"Transcription result: {transcription_result is not None}"
                        )

                    if transcription_result:
                        logging.info("Merging diarization with transcription...")
                        # Merge diarization with transcription
                        merged_segments = diarizer.merge_with_transcription(
                            diarization_result, transcription_result
                        )
                        logging.info(
                            f"Merged segments: {len(merged_segments) if merged_segments else 0}"
                        )

                        if merged_segments:
                            # Save diarized transcript
                            meeting_title = video_path.stem.replace("_", " ").replace(
                                "-", " "
                            )
                            logging.info(
                                f"Saving diarized transcript with title: {meeting_title}"
                            )
                            transcript_path = diarizer.save_diarized_transcript(
                                merged_segments, video_base_name, meeting_title
                            )
                            logging.info(f"Transcript saved to: {transcript_path}")

                            if transcript_path:
                                console.print(
                                    f"✅ [green]Diarized transcript saved[/green]"
                                )

                                # Display speaker summary
                                speaker_summary = diarizer.get_speaker_summary(
                                    diarization_result
                                )
                                if speaker_summary:
                                    console.print(f"\n👥 [cyan]Speaker Summary:[/cyan]")
                                    for speaker, stats in speaker_summary.get(
                                        "sorted_speakers", []
                                    )[:5]:
                                        duration = stats["total_duration"]
                                        segments = stats["num_segments"]
                                        console.print(
                                            f"   {speaker}: {duration:.1f}s across {segments} segments"
                                        )
                        else:
                            console.print(
                                "[yellow]⚠️  Failed to merge diarization with transcription[/yellow]"
                            )
                            logging.warning(
                                "Failed to merge diarization with transcription"
                            )
                            transcript_path = (
                                Path("transcript") / f"{video_base_name}_transcript.md"
                            )
                    else:
                        console.print(
                            "[yellow]⚠️  Failed to get transcription with timestamps[/yellow]"
                        )
                        logging.warning("Failed to get transcription with timestamps")
                        transcript_path = (
                            Path("transcript") / f"{video_base_name}_transcript.md"
                        )
                else:
                    console.print(
                        "[yellow]⚠️  Diarization failed. Continuing without speaker labels.[/yellow]"
                    )
                    logging.warning(
                        "Diarization failed, continuing without speaker labels"
                    )
                    transcript_path = (
                        Path("transcript") / f"{video_base_name}_transcript.md"
                    )
    else:
        logging.info("ENABLE_DIARIZATION flag is False, skipping diarization")
        transcript_path = Path("transcript") / f"{video_base_name}_transcript.md"

    # Clean up audio file if requested
    if CLEANUP_AUDIO:
        cleanup_audio_file(audio_path)

    # Step 7: Generate summary
    logging.info("\n[Step 7/7] Generating meeting summary...")
    summarizer = Summarizer(model_name=OLLAMA_MODEL)

    # Check if model is available, pull if not
    console.print(f"🤖 [blue]Checking for LLM model: {OLLAMA_MODEL}[/blue]")
    if not summarizer.check_model_available():
        console.print(
            f"📥 [yellow]Model '{OLLAMA_MODEL}' not found. Attempting to pull...[/yellow]"
        )
        logging.info(f"Model '{OLLAMA_MODEL}' not found. Attempting to pull...")
        if not summarizer.pull_model():
            console.print(f"❌ [red]Failed to pull model '{OLLAMA_MODEL}'![/red]")
            logging.error(f"Failed to pull model '{OLLAMA_MODEL}'")
            logging.info(
                f"Please ensure Ollama is running and the model name is correct"
            )
            return 1
        console.print(f"✅ [green]Model '{OLLAMA_MODEL}' pulled successfully![/green]")
    else:
        console.print(f"✅ [green]Model '{OLLAMA_MODEL}' is available![/green]")

    # Generate summary from transcript file
    meeting_title = video_path.stem.replace("_", " ").replace("-", " ")

    logging.info(f"Reading transcript from: {transcript_path}")
    with console.status(
        "[bold green]Generating summary with LLM...[/bold green]"
    ) as status:
        start_time = time.time()
        summary = summarizer.generate_summary(transcript_path, meeting_title)
        llm_time = time.time() - start_time
        timing_data["llm_summarization"] = llm_time

    if not summary:
        logging.error("Failed to generate summary")
        return 1

    console.print(f"✅ [green]Summary generated in {llm_time:.2f}s[/green]")

    # Save summary to file
    summary_path = construct_output_path(video_path, OUTPUT_DIR, ".md")
    if not summarizer.save_summary(summary, summary_path):
        logging.error("Failed to save summary")
        return 1

    # Calculate total time
    timing_data["total"] = sum([v for k, v in timing_data.items() if k != "total"])

    # Display beautiful summary
    display_summary(timing_data, video_path, transcript_path, summary_path)

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
