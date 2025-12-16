# test_integration.py
import sys


def test_dependencies():
    """Test that all dependencies can be imported"""
    print("Testing dependencies...")

    # Test existing dependencies
    try:
        import whisper

        print("✅ Whisper library available")
    except ImportError as e:
        print(f"❌ Whisper library not available: {e}")

    try:
        import ollama

        print("✅ Ollama library available")
    except ImportError as e:
        print(f"❌ Ollama library not available: {e}")

    try:
        import moviepy

        print("✅ MoviePy library available")
    except ImportError as e:
        print(f"❌ MoviePy library not available: {e}")

    # Test new Hugging Face dependencies
    try:
        import torch

        print(f"✅ PyTorch library available (version: {torch.__version__})")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA version: {torch.version.cuda}")
    except ImportError as e:
        print(f"❌ PyTorch library not available: {e}")

    try:
        import transformers

        print(
            f"✅ Transformers library available (version: {transformers.__version__})"
        )
    except ImportError as e:
        print(f"❌ Transformers library not available: {e}")

    try:
        import accelerate

        print(f"✅ Accelerate library available (version: {accelerate.__version__})")
    except ImportError as e:
        print(f"❌ Accelerate library not available: {e}")


def test_transcriber_classes():
    """Test that transcriber classes can be imported and initialized"""
    print("\nTesting transcriber classes...")

    try:
        from src.transcriber import Transcriber, WhisperHinglishTranscriber

        print("✅ Transcriber classes imported successfully")

        # Test base Transcriber
        try:
            transcriber = Transcriber(model_size="tiny")
            print("✅ Base Transcriber initialized successfully")
        except Exception as e:
            print(f"❌ Base Transcriber initialization failed: {e}")

        # Test WhisperHinglishTranscriber
        try:
            hinglish_transcriber = WhisperHinglishTranscriber()
            print("✅ WhisperHinglishTranscriber initialized successfully")
        except Exception as e:
            print(f"❌ WhisperHinglishTranscriber initialization failed: {e}")

    except ImportError as e:
        print(f"❌ Failed to import transcriber classes: {e}")


def test_model_loading():
    """Test that models can be loaded (without actual transcription)"""
    print("\nTesting model loading...")

    try:
        from src.transcriber import Transcriber, WhisperHinglishTranscriber

        # Test Whisper model loading
        print("Testing Whisper model loading...")
        try:
            transcriber = Transcriber(model_size="tiny")
            if transcriber.load_model():
                print("✅ Whisper model loaded successfully")
            else:
                print("❌ Whisper model loading failed")
        except Exception as e:
            print(f"❌ Whisper model loading error: {e}")

        # Test Hinglish model loading (this will download the model if not cached)
        print("\nTesting Whisper-Hinglish model loading...")
        print("Note: This may take a while on first run (model download)")
        try:
            hinglish_transcriber = WhisperHinglishTranscriber()
            if hinglish_transcriber.load_model():
                print("✅ Whisper-Hinglish model loaded successfully")
            else:
                print("❌ Whisper-Hinglish model loading failed")
        except Exception as e:
            print(f"❌ Whisper-Hinglish model loading error: {e}")

    except Exception as e:
        print(f"❌ Model loading test failed: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("INTEGRATION TEST FOR MEETINGER STT MODELS")
    print("=" * 60)

    test_dependencies()
    test_transcriber_classes()
    test_model_loading()

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
