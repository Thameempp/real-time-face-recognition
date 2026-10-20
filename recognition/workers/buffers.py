"""Latest-frame/result queue operations for the recognition worker."""

import queue


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


def read_latest_frame(
    frame_buffer,
    frame_id_value,
    frame_timestamp_value,
    frame_lock,
    last_seen_frame_id,
):
    with frame_lock:
        frame_id = frame_id_value.value

        if frame_id == last_seen_frame_id:
            return None, last_seen_frame_id

        frame = frame_buffer.copy()
        frame_timestamp = frame_timestamp_value.value

    return (
        {
            "frame_id": frame_id,
            "frame": frame,
            "frame_timestamp": frame_timestamp,
        },
        frame_id,
    )
