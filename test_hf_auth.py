#!/usr/bin/env python3
"""
Test script to verify Hugging Face authentication for pyannote models.
"""
import os
import sys
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv
from diarizer import Diarizer


def test_hf_authentication():
    """Test Hugging Face authentication with pyannote models."""
    print("=" * 60)
    print("Testing Hugging Face Authentication")
    print("=" * 60)

    # Load environment variables
    load_dotenv()

    # Get Hugging Face token
    hf_token = os.environ.get("HF_TOKEN")

    if hf_token:
        print(f"✅ HF_TOKEN found in environment")
        print(f"   Token: {hf_token[:10]}...{hf_token[-5:]}")
    else:
        print("❌ HF_TOKEN not found in environment")
        print("   Please set HF_TOKEN in .env file")
        return False

    print("\n" + "=" * 60)
    print("Attempt: Load pyannote/speaker-diarization-3.1 (gated model)")
    print("=" * 60)

    # Try with the gated model
    diarizer_v3 = Diarizer(
        model_name="pyannote/speaker-diarization-3.1", token=hf_token
    )

    success_v3 = diarizer_v3.load_model()

    if success_v3:
        print(f"✅ Successfully loaded pyannote/speaker-diarization-3.1")
        print(f"   Model: {diarizer_v3.model_name}")
        print(f"   Device: {diarizer_v3.device}")
        return True
    else:
        print(f"❌ Failed to load pyannote/speaker-diarization-3.1")
        print(f"\n📋 To fix this issue, you need to:")
        print(f"   1. Visit https://huggingface.co/pyannote/speaker-diarization-3.1")
        print(f"   2. Click 'Access repository' and accept the terms")
        print(f"   3. Visit https://huggingface.co/pyannote/segmentation-3.0")
        print(f"   4. Click 'Access repository' and accept the terms")
        print(f"   5. Wait 5-10 minutes for permissions to propagate")
        print(f"   6. Ensure HF_TOKEN is set in your .env file")
        print(f"\n💡 Note: There is no working fallback model available.")
        print(f"   You must accept the model terms to use speaker diarization.")
        return False


if __name__ == "__main__":
    try:
        success = test_hf_authentication()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
