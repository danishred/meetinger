# test_dependency_check.py
from src.utils import check_dependencies
import logging

# Set up logging to see the output
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

print("Testing dependency checker...")
result = check_dependencies()
print(f"\nDependency check result: {result}")
