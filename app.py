#!/usr/bin/env python3
"""
Meeting Summary Generator

A Python application that processes MP4 video files to generate meeting summaries.
Workflow: MP4 → Audio Extraction → VAD Processing → Transcription → Summary Generation → Markdown Output
"""

import sys
import logging
import time
import threading
from pathlib import Path
from typing import Optional, Tuple

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
    get_video_output_dir,
    check_dependencies,
)
from video_processor import extract_audio_from_video, cleanup_audio_file
from transcriber import Transcriber, WhisperHinglishTranscriber
from summarizer import Summarizer
from src.vad_processor import VADProcessor, process_audio_with_vad

# Initialize Rich console and timing tracker
console = Console()
timing_data = {
    "video_extraction": 0,
    "vad_processing": 0,
    "stt_transcription": 0,
    "llm_summarization": 0,
    "total": 0,
}

# Configuration options
USE_VAD = True  # Enable/disable Voice Activity Detection
VAD_AGGRESSIVENESS = 2  # VAD aggressiveness level (0-3, 2=moderate)
VAD_MODE = "moderate"  # VAD mode: 'conservative', 'moderate', or 'aggressive'
CREATE_VISUALIZATION = True  # Create speech activity visualization
CLEANUP_AUDIO = True  # Set to False to keep intermediate audio files


def get_video_from_input_folder() -> Optional[Path]:
    """Check input folder for video files and return the path if exactly one exists.

    Returns:
        Path to video file, None if no files found, or 'multiple_files_error' if multiple files exist
    """
    input_dir = Path("input")

    if not input_dir.exists():
        input_dir.mkdir(exist_ok=True)
        console.print("[yellow]📂 Input folder created (empty)[/yellow]")
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

    if not video_files:
        console.print(
            "[yellow]📂 Input folder is empty. Will prompt for video file.[/yellow]"
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


class CountdownInput:
    """Handles countdown timer for model selection with automatic fallback.

    This class provides a thread-safe way to get user input with a countdown timer.
    If no input is received within the timeout period, it automatically selects
    a default option.
    """

    def __init__(self, timeout_seconds: int = 10):
        """Initialize the countdown input handler.

        Args:
            timeout_seconds: Number of seconds to wait for user input before auto-selecting
        """
        self.timeout_seconds = timeout_seconds
        self.user_input = None
        self.input_received = threading.Event()
        self.countdown_thread = None
        self.input_thread = None
        self._lock = threading.Lock()
        logging.debug(f"CountdownInput initialized with timeout: {timeout_seconds}s")

    def _countdown_timer(self) -> None:
        """Display countdown and auto-select if no input received."""
        logging.debug(f"Countdown thread started, timeout: {self.timeout_seconds}s")
        try:
            for i in range(self.timeout_seconds, 0, -1):
                if self.input_received.is_set():
                    logging.debug("Countdown thread: Input received, exiting early")
                    return

                print(
                    f"\r⏳ Auto-selecting in {i} seconds... Press Enter to choose now: ",
                    end="",
                    flush=True,
                )
                time.sleep(1)

            # Time's up - auto-select first option
            if not self.input_received.is_set():
                with self._lock:
                    logging.debug(
                        "Countdown thread: Time's up, auto-selecting option 1"
                    )
                    print(
                        f"\r⏰ Time's up! Auto-selecting first option (Whisper-Hinglish)..."
                    )
                    self.user_input = "1"
                    self.input_received.set()
                    logging.debug("Countdown thread: Auto-selection completed")

        except Exception as e:
            logging.error(f"Countdown thread error: {e}")
        finally:
            logging.debug("Countdown thread: Exiting")

    def _input_reader(self) -> None:
        """Read user input in a separate thread."""
        try:
            logging.debug("_input_reader: Waiting for user input...")
            user_choice = input().strip()
            logging.debug(f"_input_reader: User input received: '{user_choice}'")

            with self._lock:
                self.user_input = user_choice
                self.input_received.set()
                logging.debug("_input_reader: Input event set")

        except (EOFError, KeyboardInterrupt):
            logging.debug(
                "_input_reader: Input exception caught (EOF/KeyboardInterrupt)"
            )
            # If input fails, auto-select first option
            if not self.input_received.is_set():
                with self._lock:
                    self.user_input = "1"
                    self.input_received.set()
                    logging.debug(
                        "_input_reader: Auto-selected option 1 due to input exception"
                    )
        except Exception as e:
            logging.error(f"_input_reader error: {e}")
        finally:
            logging.debug("_input_reader: Exiting")

    def get_input_with_countdown(self, prompt: str) -> str:
        """Get user input with a countdown timer.

        Args:
            prompt: The prompt message to display to the user

        Returns:
            The user's choice ('1' or '2') or auto-selected choice after timeout
        """
        logging.debug("get_input_with_countdown: Starting")
        print(prompt)

        # Start countdown in background
        self.countdown_thread = threading.Thread(target=self._countdown_timer)
        self.countdown_thread.daemon = True
        self.countdown_thread.start()
        logging.debug("get_input_with_countdown: Countdown thread started")

        # Start input reader in background
        self.input_thread = threading.Thread(target=self._input_reader)
        self.input_thread.daemon = True
        self.input_thread.start()
        logging.debug("get_input_with_countdown: Input reader thread started")

        # Wait for either input or timeout
        logging.debug("get_input_with_countdown: Waiting for input or timeout...")
        self.input_received.wait(timeout=self.timeout_seconds + 1)

        # Ensure both threads are finished
        logging.debug("get_input_with_countdown: Waiting for threads to finish...")
        if self.countdown_thread and self.countdown_thread.is_alive():
            self.countdown_thread.join(timeout=0.1)
        if self.input_thread and self.input_thread.is_alive():
            self.input_thread.join(timeout=0.1)

        # Ensure we have a valid input
        with self._lock:
            if self.user_input is None:
                logging.debug(
                    "get_input_with_countdown: No input received, using default"
                )
                self.user_input = "1"
                self.input_received.set()

        logging.debug(
            f"get_input_with_countdown: Returning user_input: '{self.user_input}'"
        )
        return self.user_input


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


def _select_stt_model() -> Tuple[object, str]:
    """Handle STT model selection with countdown timer.

    Returns:
        Tuple of (transcriber_instance, model_name)
    """
    print("\n" + "=" * 60)
    print("Choose STT Model:")
    print("=" * 60)
    print(
        "1. Whisper-Hinglish (Oriserve) - Specialized for Hinglish (Latin script output)"
    )
    print("2. Whisper (OpenAI) - Multi-language, general purpose")
    print("=" * 60)

    # Auto-select model for non-interactive environments
    if sys.stdin.isatty():
        # Interactive mode - prompt user with countdown
        countdown_input = CountdownInput(timeout_seconds=10)
        choice = countdown_input.get_input_with_countdown("Enter choice (1 or 2): ")
    else:
        # Non-interactive mode - auto-select
        choice = "1"
        print("Auto-selecting Whisper-Hinglish model (non-interactive mode)")

    if choice == "1":
        transcriber = WhisperHinglishTranscriber()
        model_name = "Whisper-Hinglish (Oriserve)"
    elif choice == "2":
        if sys.stdin.isatty():
            model_size = input(
                "Enter Whisper model size (tiny, base, small, medium, large) [medium]: "
            ).strip()
        else:
            model_size = "medium"
        if not model_size:
            model_size = "medium"
        transcriber = Transcriber(model_size=model_size)
        model_name = f"Whisper ({model_size})"
    else:
        print("Invalid choice. Defaulting to Whisper-Hinglish (Oriserve)")
        transcriber = WhisperHinglishTranscriber()
        model_name = "Whisper-Hinglish (Oriserve)"

    console.print(f"🤖 [blue]Selected Model:[/blue] {model_name}")
    return transcriber, model_name


def _find_video_file() -> Optional[Path]:
    """Find and return the video file to process.

    Returns:
        Path to video file, or None if not found
    """
    # Check for video in input folder
    video_path = get_video_from_input_folder()

    if video_path == "multiple_files_error":
        console.print("\n[red]Exiting due to multiple video files.[/red]")
        return None
    elif video_path:
        # Use the video from input folder
        console.print(f"🎬 [cyan]Processing:[/cyan] {video_path.name}")
        logging.info(f"Processing: {video_path.name}")
        return video_path
    else:
        # Fall back to Windows Videos folder - get most recent video
        VIDEO_DIR = "/mnt/c/Users/danis/Videos"  # Windows Videos folder in WSL
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
            return None

        console.print(
            f"✅ [green]Video loaded successfully from Windows Videos folder![/green] {video_path.name}"
        )
        logging.info(f"Processing: {video_path.name}")
        return video_path


def _process_audio_pipeline(
    video_path: Path, video_output_dir: Path, transcriber: object
) -> Optional[Path]:
    """Process the audio pipeline: extraction → VAD → transcription.

    Args:
        video_path: Path to input video file
        video_output_dir: Output directory for processed files
        transcriber: Transcriber instance to use

    Returns:
        Path to audio file ready for transcription, or None if failed
    """
    # Step 1: Extract audio from video
    logging.info("\n[Step 4/7] Extracting audio from video...")
    console.print("🎵 [blue]Extracting audio from video...[/blue]")
    start_time = time.time()
    audio_path = extract_audio_from_video(video_path, str(video_output_dir))
    timing_data["video_extraction"] = time.time() - start_time

    if not audio_path:
        logging.error("Failed to extract audio from video")
        return None

    console.print(
        f"✅ [green]Audio extracted in {timing_data['video_extraction']:.2f}s[/green]"
    )

    # Step 2: Process audio with VAD (if enabled)
    audio_to_transcribe = audio_path
    filtered_audio_path = None

    if USE_VAD:
        logging.info("\n[Step 5/7] Processing audio with Voice Activity Detection...")
        console.print("🔊 [blue]Applying Voice Activity Detection...[/blue]")

        start_time = time.time()
        filtered_audio_path, total_speech_duration = process_audio_with_vad(
            audio_path=str(audio_path),
            mode=VAD_MODE,
            aggressiveness=VAD_AGGRESSIVENESS,
            max_silence_duration=1.5,
            min_speech_duration=0.5,
            output_dir=str(video_output_dir),
        )
        vad_time = time.time() - start_time
        timing_data["vad_processing"] = vad_time

        if filtered_audio_path:
            audio_to_transcribe = filtered_audio_path
            console.print(
                f"✅ [blue]VAD processing completed in {vad_time:.2f}s[/blue]"
            )
            console.print(f"🎯 [blue]Filtered audio ready for transcription[/blue]")
            console.print(
                f"📊 [blue]Total speech duration: {total_speech_duration:.2f}s[/blue]"
            )
        else:
            console.print(
                "⚠️ [yellow]VAD processing failed, using original audio[/yellow]"
            )

    # Step 3: Transcribe audio
    logging.info("\n[Step 6/7] Transcribing audio...")
    console.print(f"📦 [blue]Loading {type(transcriber).__name__}...[/blue]")

    if not transcriber.load_model():
        console.print(f"❌ [red]Failed to load {type(transcriber).__name__}![/red]")
        logging.error(f"Failed to load {type(transcriber).__name__}")
        if CLEANUP_AUDIO:
            cleanup_audio_file(audio_path)
            if USE_VAD and filtered_audio_path and filtered_audio_path != audio_path:
                cleanup_audio_file(filtered_audio_path)
        return None

    console.print(
        f"✅ [green]{type(transcriber).__name__} loaded successfully![/green]"
    )

    # Transcribe audio and save to transcript folder
    video_base_name = video_path.stem
    logging.info(f"Starting transcription of {audio_to_transcribe}")
    console.print("[bold green]Transcribing audio...[/bold green]")

    start_time = time.time()
    # Convert string path to Path object if needed
    if isinstance(audio_to_transcribe, str):
        audio_to_transcribe = Path(audio_to_transcribe)

    transcription = transcriber.transcribe_audio(
        audio_to_transcribe, video_base_name, str(video_output_dir)
    )
    stt_time = time.time() - start_time
    timing_data["stt_transcription"] = stt_time

    if not transcription:
        logging.error("Failed to transcribe audio")
        if CLEANUP_AUDIO:
            cleanup_audio_file(audio_path)
            if USE_VAD and filtered_audio_path and filtered_audio_path != audio_path:
                cleanup_audio_file(filtered_audio_path)
        return None

    console.print(f"✅ [green]Transcription completed in {stt_time:.2f}s[/green]")
    logging.info(f"Transcription preview (first 200 chars): {transcription[:200]}...")

    # Clean up audio file if requested
    if CLEANUP_AUDIO:
        cleanup_audio_file(audio_path)

    return audio_to_transcribe


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
    OUTPUT_DIR = "output"
    OLLAMA_MODEL = "gemma2:9b-instruct-q4_K_M"

    print("=" * 60)

    # Step 1: Check dependencies
    logging.info("\n[Step 1/7] Checking dependencies...")
    console.print("🔍 [blue]Checking dependencies...[/blue]")
    if not check_dependencies():
        console.print("❌ [red]Dependency check failed![/red]")
        logging.error("Dependency check failed. Please install required dependencies.")
        logging.info("Run: pip install -r requirements.txt")
        logging.info("Also ensure ffmpeg is installed and Ollama is running")
        return 1
    console.print("✅ [green]All dependencies satisfied![/green]")

    # Step 2: Get video file
    logging.info("\n[Step 2/7] Finding video file...")
    console.print("🎥 [blue]Finding video file...[/blue]")
    video_path = _find_video_file()

    if video_path is None:
        return 1

    # Check file size and warn if large
    video_size_mb = video_path.stat().st_size / (1024 * 1024)
    if video_size_mb > 100:
        console.print(
            f"⚠️ [yellow]Warning: Large video file detected ({video_size_mb:.1f} MB)[/yellow]"
        )
        console.print(f"   [dim]Processing may take longer than usual.[/dim]")

    # Step 3: Create video-specific output directory
    logging.info("\n[Step 3/7] Preparing video-specific output directory...")
    video_output_dir = get_video_output_dir(video_path, OUTPUT_DIR)
    logging.info(f"Using output directory: {video_output_dir}")

    # Step 4-6: Process audio pipeline
    transcriber, model_name = _select_stt_model()
    audio_to_transcribe = _process_audio_pipeline(
        video_path, video_output_dir, transcriber
    )

    if audio_to_transcribe is None:
        return 1

    # Step 7: Generate summary
    logging.info("\n[Step 7/7] Generating meeting summary...")
    console.print("🤖 [blue]Generating meeting summary...[/blue]")

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
    transcript_path = (
        Path(video_output_dir) / "transcript" / f"{video_path.stem}_transcript.md"
    )

    logging.info(f"Reading transcript from: {transcript_path}")
    console.print("[bold green]Generating summary with LLM...[/bold green]")

    start_time = time.time()
    summary = summarizer.generate_summary(
        transcript_path, meeting_title, str(video_output_dir)
    )
    llm_time = time.time() - start_time
    timing_data["llm_summarization"] = llm_time

    if not summary:
        logging.error("Failed to generate summary")
        return 1

    console.print(f"✅ [green]Summary generated in {llm_time:.2f}s[/green]")
    logging.debug(f"Summary generation completed, duration: {llm_time:.2f}s")

    # Save summary to file
    summary_path = construct_output_path(video_path, str(video_output_dir), ".md")
    if not summarizer.save_summary(summary, summary_path):
        logging.error("Failed to save summary")
        return 1

    # Calculate total time and display summary
    timing_data["total"] = sum([v for k, v in timing_data.items() if k != "total"])
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
