"""Video writing utility — atomic mp4 from a list of RGB frames."""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def write_mp4(frames: list[np.ndarray], path: Path | str, fps: int = 30) -> None:
    """Write *frames* to *path* as an mp4, atomically (tmp → rename).

    Parameters
    ----------
    frames
        Non-empty list of RGB uint8 arrays, all same H×W×3.
    path
        Destination .mp4 path. Parent directory must exist.
    fps
        Output framerate. 30 matches Genesis native.

    Raises
    ------
    ValueError
        If *frames* is empty.
    """
    if not frames:
        raise ValueError("frames must be non-empty")
    path = Path(path)
    # Keep .mp4 extension on tmp path — imageio picks codec by extension.
    tmp = path.with_suffix(".tmp" + path.suffix)
    try:
        with imageio.get_writer(str(tmp), fps=fps, codec="libx264") as writer:
            for f in frames:
                writer.append_data(f)
        tmp.replace(path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


__all__ = ["write_mp4"]
