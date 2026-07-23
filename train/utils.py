import queue
import re


def clean_name(name):
    """Convert a person's name into a safe folder name."""
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9_-]+", "_", name)

    return name.strip("_")


def put_latest(target_queue, value):
    try:
        target_queue.put_nowait(value)
        return
    except queue.Full:
        pass

    try:
        target_queue.get_nowait()
    except queue.Empty:
        pass

    try:
        target_queue.put_nowait(value)
    except queue.Full:
        pass
