#!/usr/bin/env python3
"""
Test script to verify the enhanced VAD functionality.
"""

import sys
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from vad_processor import VADProcessor, process_audio_with_vad


def test_vad_processor():
    """Test the enhanced VAD processor functionality."""
    print("🧪 Testing Enhanced VAD Processor...")

    # Test VADProcessor initialization
    try:
        vad_processor = VADProcessor(aggressiveness=3)
        print("✅ VADProcessor initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize VADProcessor: {e}")
        return False

    # Test process_audio_with_vad function signature
    try:
        # This would normally require an actual audio file
        # but we can at least test the function signature
        import inspect

        sig = inspect.signature(process_audio_with_vad)
        params = list(sig.parameters.keys())
        expected_params = [
            "audio_path",
            "aggressiveness",
            "max_silence_duration",
            "min_speech_duration",
            "output_dir",
        ]

        if all(param in params for param in expected_params):
            print("✅ process_audio_with_vad has correct signature")
        else:
            print(
                f"❌ process_audio_with_vad signature mismatch. Expected: {expected_params}, Got: {params}"
            )
            return False

    except Exception as e:
        print(f"❌ Failed to test process_audio_with_vad signature: {e}")
        return False

    # Test VADProcessor methods
    try:
        # Test get_speech_statistics method exists
        if hasattr(vad_processor, "get_speech_statistics"):
            print("✅ get_speech_statistics method exists")
        else:
            print("❌ get_speech_statistics method missing")
            return False

        # Test get_vad_segments method exists
        if hasattr(vad_processor, "get_vad_segments"):
            print("✅ get_vad_segments method exists")
        else:
            print("❌ get_vad_segments method missing")
            return False

    except Exception as e:
        print(f"❌ Failed to test VADProcessor methods: {e}")
        return False

    print("🎉 All VAD enhancement tests passed!")
    return True


if __name__ == "__main__":
    success = test_vad_processor()
    sys.exit(0 if success else 1)
