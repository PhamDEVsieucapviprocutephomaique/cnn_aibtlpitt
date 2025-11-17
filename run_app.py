"""
Run Application - Simple launcher
"""
import os
import sys

# Add to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.main_app import main

if __name__ == "__main__":
    print("Starting Neural Recognition App...")
    main()