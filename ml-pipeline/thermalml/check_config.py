"""python -m thermalml.check_config [config.yaml]: list values the team still has to fix."""
import sys

from .config import main

if __name__ == "__main__":
    sys.exit(main())
