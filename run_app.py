"""
Run Application Script
Chạy file này để launch ứng dụng
"""
import os
import sys


# Add to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.main_app import main

if __name__ == "__main__":
    print("=" * 60)
    print("NEURAL RECOGNITION SYSTEM")
    print("Starting application...")
    print("=" * 60)
    main()