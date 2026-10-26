import sys
from pathlib import Path

# Add project root and src to sys.path
root = Path(__file__).parent.parent
sys.path.insert(0, str(root / "src"))

from konnon_draw.web_server import main

if __name__ == "__main__":
    main()
