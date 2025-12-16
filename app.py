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

# Initialize Rich console and timing tracker
console = Console()
timing_data = {
    "video_extraction": 0,
    "stt_transcription": 0,
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
        # Fall back to current implementation - ask user for file
        console.print("[cyan]Enter the path to the video file:[/cyan]")
        video_path_str = input("> ").strip()

        if not video_path_str:
            console.print("[yellow]No path provided. Exiting.[/yellow]")
            return 0

        video_path = Path(video_path_str)

        if not video_path.exists():
            console.print(f"[red]❌ File not found: {video_path}[/red]")
            return 1

        if not video_path.is_file():
            console.print(f"[red]❌ Path is not a file: {video_path}[/red]")
            return 1

        console.print(f"🎬 [cyan]Processing:[/cyan] {video_path.name}")
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

    # Clean up audio file if requested
    if CLEANUP_AUDIO:
        cleanup_audio_file(audio_path)

    # Step 6: Generate summary
    logging.info("\n[Step 6/6] Generating meeting summary...")
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
    transcript_path = Path("transcript") / f"{video_base_name}_transcript.md"

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
