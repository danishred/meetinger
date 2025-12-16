#!/usr/bin/env python3
"""
Test script to verify the hanging issue is fixed.
This simulates the original problem and verifies the fix.
"""

import sys
import time
import threading
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_original_hanging_behavior():
    """
    Simulate the original hanging behavior to show the problem was real.
    This would hang indefinitely in the original code.
    """
    print("Testing original hanging behavior simulation...")
    print("This would hang in the original code after countdown completion.")

    # Simulate the original problematic code pattern
    input_received = threading.Event()
    user_input = None

    def countdown_timer():
        """Original countdown logic"""
        for i in range(3, 0, -1):
            if input_received.is_set():
                return
            print(
                f"\r⏳ Original: Auto-selecting in {i} seconds...", end="", flush=True
            )
            time.sleep(1)

        if not input_received.is_set():
            print(f"\r⏰ Time's up! Auto-selecting option 1...")
            user_input = "1"  # This assignment is local, not affecting outer scope!
            input_received.set()

    def get_input_original():
        """Original problematic input handling"""
        nonlocal user_input
        countdown_thread = threading.Thread(target=countdown_timer)
        countdown_thread.daemon = True
        countdown_thread.start()

        # This is the problem: input() blocks indefinitely
        try:
            user_choice = input().strip()  # This would hang!
            user_input = user_choice
            input_received.set()
        except:
            if not input_received.is_set():
                user_input = "1"
                input_received.set()

        countdown_thread.join(timeout=0.1)
        return user_input

    # This would hang in the original code
    start_time = time.time()
    try:
        # Use a timeout to prevent actual hanging in test
        result = None

        def run_input():
            nonlocal result
            result = get_input_original()

        input_thread = threading.Thread(target=run_input)
        input_thread.daemon = True
        input_thread.start()
        input_thread.join(timeout=4)  # 4 second timeout

        elapsed = time.time() - start_time
        if input_thread.is_alive():
            print(
                f"❌ Original code would HANG (thread still running after {elapsed:.2f}s)"
            )
            return False
        else:
            print(f"✅ Original code completed in {elapsed:.2f}s")
            return True
    except Exception as e:
        print(f"❌ Original code failed with error: {e}")
        return False


def test_fixed_behavior():
    """Test the fixed behavior"""
    print("\nTesting fixed behavior...")

    from app import CountdownInput

    # Test with very short timeout to avoid waiting
    countdown = CountdownInput(timeout_seconds=2)

    start_time = time.time()
    choice = countdown.get_input_with_countdown("Testing fixed version: ")
    elapsed_time = time.time() - start_time

    print(f"Choice: '{choice}', Time: {elapsed_time:.2f}s")

    # The fix should complete within timeout + small buffer
    if choice == "1" and elapsed_time <= 3.0:
        print("✅ Fixed version works correctly!")
        return True
    else:
        print("❌ Fixed version has issues")
        return False


if __name__ == "__main__":
    print("=" * 70)
    print("Testing Hanging Issue Fix")
    print("=" * 70)

    # Test 1: Show original problem existed
    print("\n1. Testing original hanging behavior...")
    original_had_issue = not test_original_hanging_behavior()

    # Test 2: Show fix works
    print("\n2. Testing fixed behavior...")
    fix_works = test_fixed_behavior()

    print("\n" + "=" * 70)
    print("SUMMARY:")
    if original_had_issue:
        print("✅ Confirmed: Original code had hanging issue")
    else:
        print("ℹ️  Note: Could not reproduce original hanging in test environment")

    if fix_works:
        print("✅ Confirmed: Fix resolves the hanging issue")
    else:
        print("❌ Issue: Fix does not work properly")

    if fix_works:
        print("\n🎉 SUCCESS: The hanging issue has been resolved!")
    else:
        print("\n❌ FAILURE: The hanging issue persists")
    print("=" * 70)
