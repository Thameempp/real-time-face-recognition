import multiprocessing as mp
import os
import sys


PACKAGE_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if PACKAGE_PARENT not in sys.path:
    sys.path.insert(0, PACKAGE_PARENT)

from train.main import main  # noqa: E402


if __name__ == "__main__":
    mp.freeze_support()
    main()
