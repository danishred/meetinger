#!/usr/bin/env python3
"""
Test script to diagnose diarization flow issues.
This script tests the complete diarization workflow with detailed logging.
"""

import sys
import logging
import os
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils import setup_logging
from diarizer import Diarizer


def test_hugging_face_token():
    """Test if Hugging Face token is configured."""
    print("\n" + "=" * 60)
    print("Testing Hugging Face Token Configuration")
    print("=" * 60)

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        print(f"✅ HF_TOKEN is set in environment")
        print(f"   Token length: {len(hf_token)} characters")
        print(f"   Token starts with: {hf_token[:10]}...")
    else:
        print("❌ HF_TOKEN is NOT set in environment")
        print("   This will cause authentication errors with pyannote models")
        print("   To set the token, run:")
        print("   export HF_TOKEN='your_token_here'")
        print("   Or add it to your ~/.bashrc or ~/.zshrc file")

    return hf_token is not None


def test_diarizer_initialization():
    """Test diarizer initialization with and without token."""
    print("\n" + "=" * 60)
    print("Testing Diarizer Initialization")
    print("=" * 60)

    hf_token = os.environ.get("HF_TOKEN")

    try:
        if hf_token:
            print(f"Initializing Diarizer with HF_TOKEN...")
            diarizer = Diarizer(use_auth_token=hf_token)
        else:
            print(f"Initializing Diarizer without HF_TOKEN...")
            diarizer = Diarizer()

        print(f"✅ Diarizer initialized successfully")
        print(f"   Device: {diarizer.device}")
        print(f"   Model: {diarizer.model_name}")
        print(f"   Auth token set: {'Yes' if diarizer.use_auth_token else 'No'}")
        return diarizer
    except Exception as e:
        print(f"❌ Failed to initialize diarizer: {e}")
        import traceback

        traceback.print_exc()
        return None


def test_diarizer_model_loading(diarizer):
    """Test diarizer model loading with detailed logging."""
    print("\n" + "=" * 60)
    print("Testing Model Loading")
    print("=" * 60)

    try:
        print("Loading model...")
        success = diarizer.load_model()

        if success:
            print("✅ Model loaded successfully")
            print(f"   Pipeline: {diarizer.pipeline}")
            print(f"   Device: {diarizer.device}")
            return True
        else:
            print("❌ Model loading failed")
            print("   Check the logs above for detailed error messages")
            return False
    except Exception as e:
        print(f"❌ Exception during model loading: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_diarization_with_sample_audio(diarizer, audio_path):
    """Test diarization with a sample audio file."""
    print("\n" + "=" * 60)
    print("Testing Diarization with Sample Audio")
    print("=" * 60)

    if not audio_path.exists():
        print(f"❌ Audio file not found: {audio_path}")
        return None

    print(f"Audio file: {audio_path}")
    print(f"File size: {audio_path.stat().st_size / (1024*1024):.2f} MB")

    try:
        print("Running diarization...")
        result = diarizer.diarize(audio_path)

        if result:
            print("✅ Diarization successful!")
            print(f"   Segments: {len(result['segments'])}")
            print(f"   Speakers: {result['num_speakers']}")
            print(f"   Duration: {result['total_duration']:.1f}s")

            # Show first few segments
            if result["segments"]:
                print("\nFirst 5 segments:")
                for i, seg in enumerate(result["segments"][:5]):
                    print(
                        f"   {i+1}. [{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['speaker']}"
                    )

            return result
        else:
            print("❌ Diarization failed")
            print("   Check the logs above for detailed error messages")
            return None
    except Exception as e:
        print(f"❌ Exception during diarization: {e}")
        import traceback

        traceback.print_exc()
        return None


def check_pyannote_dependencies():
    """Check if pyannote dependencies are installed."""
    print("\n" + "=" * 60)
    print("Checking Pyannote Dependencies")
    print("=" * 60)

    dependencies = {
        "pyannote.audio": False,
        "pyannote.core": False,
        "librosa": False,
        "soundfile": False,
    }

    for dep in dependencies:
        try:
            __import__(dep)
            dependencies[dep] = True
            print(f"✅ {dep} is installed")
        except ImportError:
            print(f"❌ {dep} is NOT installed")

    all_installed = all(dependencies.values())
    if all_installed:
        print("\n✅ All pyannote dependencies are installed")
    else:
        print("\n❌ Some pyannote dependencies are missing")
        print("   Install them with:")
        print("   pip install pyannote.audio pyannote.core librosa soundfile")

    return all_installed


def main():
    """Run comprehensive diarization flow tests."""
    print("=" * 60)
    print("Diarization Flow Diagnostic Test")
    print("=" * 60)

    # Set up logging
    setup_logging(logging.INFO)
    logging.info("Starting diarization flow diagnostic test")

    # Test 1: Check dependencies
    deps_ok = check_pyannote_dependencies()

    # Test 2: Check HF Token
    token_ok = test_hugging_face_token()

    # Test 3: Initialize diarizer
    diarizer = test_diarizer_initialization()

    if not diarizer:
        print("\n" + "=" * 60)
        print("❌ Cannot proceed - diarizer initialization failed")
        print("=" * 60)
        return 1

    # Test 4: Load model
    model_loaded = test_diarizer_model_loading(diarizer)

    if not model_loaded:
        print("\n" + "=" * 60)
        print("❌ Cannot proceed - model loading failed")
        print("=" * 60)
        return 1

    # Test 5: Check for sample audio files
    print("\n" + "=" * 60)
    print("Looking for Sample Audio Files")
    print("=" * 60)

    # Check common locations
    audio_locations = [
        Path("output") / "2025-12-16 09-57-46.wav",  # From existing output
        Path("output") / "test_audio.wav",
        Path("input") / "test.mp4",
    ]

    found_audio = None
    for audio_path in audio_locations:
        if audio_path.exists():
            found_audio = audio_path
            print(f"✅ Found audio file: {audio_path}")
            break

    if not found_audio:
        print("❌ No audio files found for testing")
        print("   To test diarization, you need to:")
        print("   1. Place a video file in the 'input' folder")
        print("   2. Run app.py to extract audio")
        print("   3. Then run this test again")
        return 1

    # Test 6: Run diarization
    result = test_diarization_with_sample_audio(diarizer, found_audio)

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    results = {
        "Dependencies": "✅" if deps_ok else "❌",
        "HF Token": "✅" if token_ok else "❌",
        "Initialization": "✅" if diarizer else "❌",
        "Model Loading": "✅" if model_loaded else "❌",
        "Diarization": "✅" if result else "❌",
    }

    for test, status in results.items():
        print(f"{status} {test}")

    all_passed = all(status == "✅" for status in results.values())

    if all_passed:
        print("\n✅ All tests passed! Diarization is working correctly.")
        return 0
    else:
        print(f"\n⚠️  {sum(1 for s in results.values() if s == '❌')} test(s) failed")
        print("   Check the output above for details on what needs to be fixed")
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
