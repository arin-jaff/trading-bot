"""Launch the Streamlit dashboard."""

import subprocess
import sys


def main():
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        "src/gui/dashboard.py",
        "--server.port", "8501",
        "--server.headless", "true",
    ])


if __name__ == "__main__":
    main()
