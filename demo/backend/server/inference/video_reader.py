"""
Module for reading frames from video files using OpenCV.
"""

import logging
import cv2
from pathlib import Path
from typing import List
import numpy as np

logger = logging.getLogger(__name__)

def read_raw_frames(relative_path: str, data_path: Path) -> List[np.ndarray]:
    """
    Read raw frames from a video file using OpenCV.

    Args:
        relative_path (str): Relative path to the video (e.g., 'gallery/video.mp4').
        data_path (Path): The base data directory path.

    Returns:
        List[np.ndarray]: List of frames (as BGR NumPy arrays) read from the video.

    Raises:
        FileNotFoundError: If the video file does not exist.
        RuntimeError: If the video file cannot be opened or read.
    """
    video_path = data_path / relative_path
    if not video_path.exists():
        logger.error(f"Video file not found at {video_path}")
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Could not open video at {video_path}")
        raise RuntimeError(f"Could not open video file: {video_path}")

    frames = []
    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            frame_count += 1
    except Exception as e:
        logger.error(f"Error reading frames from video {video_path}: {e}", exc_info=True)
        raise RuntimeError(f"Error reading video file {video_path}: {e}") from e
    finally:
        cap.release()

    logger.info(f"Loaded {frame_count} raw frames from {video_path}")
    if not frames:
         logger.warning(f"Video file {video_path} was opened but no frames were read.")

    return frames