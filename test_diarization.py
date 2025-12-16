#!/usr/bin/env python3
"""
Test script for speaker diarization functionality.
This script tests the basic functionality of the Diarizer class.
"""

import sys
import logging
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils import setup_logging
from diarizer import Diarizer


def test_diarizer_initialization():
    """Test diarizer initialization."""
    print("Testing Diarizer initialization...")
    try:
        diarizer = Diarizer()
        print("✅ Diarizer initialized successfully")
        print(f"   Device: {diarizer.device}")
        print(f"   Model: {diarizer.model_name}")
        return diarizer
    except Exception as e:
        print(f"❌ Failed to initialize diarizer: {e}")
        return None


def test_diarizer_model_loading(diarizer):
    """Test diarizer model loading."""
    print("\nTesting model loading...")
    try:
        if diarizer.load_model():
            print("✅ Model loaded successfully")
            return True
        else:
            print("❌ Failed to load model")
            return False
    except Exception as e:
        print(f"❌ Error during model loading: {e}")
        return False


def test_diarizer_formatting():
    """Test markdown formatting functionality."""
    print("\nTesting markdown formatting...")
    try:
        diarizer = Diarizer()

        # Create sample merged segments
        sample_segments = [
            {
                "start": 0.0,
                "end": 5.0,
                "speaker": "SPEAKER_00",
                "text": "Hello everyone, welcome to the meeting.",
            },
            {
                "start": 5.5,
                "end": 10.0,
                "speaker": "SPEAKER_01",
                "text": "Thank you for having me.",
            },
            {
                "start": 10.5,
                "end": 15.0,
                "speaker": "SPEAKER_00",
                "text": "Let's get started with the agenda.",
            },
        ]

        # Test formatting
        formatted = diarizer.format_diarized_markdown(sample_segments, "Test Meeting")
        print("✅ Markdown formatting successful")
        print("\nFormatted output preview:")
        print("-" * 50)
        print(formatted[:200] + "..." if len(formatted) > 200 else formatted)
        print("-" * 50)
        return True
    except Exception as e:
        print(f"❌ Error during formatting test: {e}")
        return False


def test_speaker_summary():
    """Test speaker summary generation."""
    print("\nTesting speaker summary generation...")
    try:
        diarizer = Diarizer()

        # Create sample diarization result
        sample_result = {
            "segments": [
                {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00", "duration": 5.0},
                {"start": 5.5, "end": 10.0, "speaker": "SPEAKER_01", "duration": 4.5},
                {"start": 10.5, "end": 15.0, "speaker": "SPEAKER_00", "duration": 4.5},
            ],
            "num_speakers": 2,
            "total_duration": 15.0,
        }

        # Test summary generation
        summary = diarizer.get_speaker_summary(sample_result)
        print("✅ Speaker summary generation successful")
        print(f"   Total speakers: {summary.get('total_speakers', 0)}")

        sorted_speakers = summary.get("sorted_speakers", [])
        for speaker, stats in sorted_speakers:
            duration = stats["total_duration"]
            segments = stats["num_segments"]
            print(f"   {speaker}: {duration:.1f}s across {segments} segments")

        return True
    except Exception as e:
        print(f"❌ Error during speaker summary test: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Speaker Diarization Test Suite")
    print("=" * 60)

    # Set up logging
    setup_logging(logging.INFO)

    # Run tests
    tests_passed = 0
    total_tests = 4

    # Test 1: Initialization
    diarizer = test_diarizer_initialization()
    if diarizer:
        tests_passed += 1

    # Test 2: Model Loading (only if initialization succeeded)
    if diarizer:
        if test_diarizer_model_loading(diarizer):
            tests_passed += 1
    else:
        print("\n⚠️  Skipping model loading test due to initialization failure")
        total_tests -= 1

    # Test 3: Formatting
    if test_diarizer_formatting():
        tests_passed += 1

    # Test 4: Speaker Summary
    if test_speaker_summary():
        tests_passed += 1

    # Summary
    print("\n" + "=" * 60)
    print(f"Test Results: {tests_passed}/{total_tests} tests passed")
    print("=" * 60)

    if tests_passed == total_tests:
        print("✅ All tests passed!")
        return 0
    else:
        print(f"⚠️  {total_tests - tests_passed} test(s) failed")
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
