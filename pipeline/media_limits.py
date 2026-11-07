"""Shared video resource policy. Audio retains its separate 30-second limit."""
VIDEO_MAX_BYTES = 10 * 1024 * 1024
VIDEO_MAX_SECONDS = 15
VIDEO_MAX_LONG_EDGE = 1920
VIDEO_MAX_SHORT_EDGE = 1080
VIDEO_MAX_FPS = 30


def dimensions_ok(width, height):
    return (0 < min(width, height) <= VIDEO_MAX_SHORT_EDGE
            and max(width, height) <= VIDEO_MAX_LONG_EDGE)
