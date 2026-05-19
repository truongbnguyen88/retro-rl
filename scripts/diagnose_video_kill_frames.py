"""Extract video frames around a kill event for visual inspection.

The eval video is one rendered frame per outer policy step (action_repeat=4
emulator frames), saved at 30 fps. CSV step N from
``diagnose_eval_trace.py`` corresponds to video frame N.

Usage:
    python scripts/diagnose_video_kill_frames.py \\
        --video outputs/videos/ppo_airstriker_v5/eval-step-100000.mp4 \\
        --trace outputs/diagnostics/v5_100k_eval_trace.csv \\
        --kills 1 2 \\
        --before 8 --after 24 \\
        --out outputs/diagnostics/frames_100k

For each selected kill, writes a sequence of PNGs labelled with their CSV
step and score so we can flip through them and check whether bullet sprites
remain visible on the screen below the explosion (-> occlusion hypothesis)
or vanish entirely (-> bullet-cap hypothesis).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2


def load_kills(trace_csv: Path) -> list[dict]:
    kills: list[dict] = []
    with open(trace_csv) as f:
        for row in csv.DictReader(f):
            if int(row["score_delta"]) > 0:
                kills.append(
                    {
                        "step": int(row["step"]),
                        "score": int(row["score"]),
                        "score_delta": int(row["score_delta"]),
                        "game_sec": float(row["game_sec"]),
                        "lives_dropped": row["lives_dropped"] == "YES",
                    }
                )
    return kills


def extract_window(
    video_path: Path,
    center_frame: int,
    before: int,
    after: int,
    out_dir: Path,
    label: str,
) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start = max(0, center_frame - before)
    end = min(total, center_frame + after + 1)

    # Seek to start. mp4 keyframe seeking can be approximate; we compensate by
    # reading sequentially from frame 0 if the requested window is near the
    # start, which is the common case for early kills.
    if start < 60:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for _ in range(start):
            cap.read()
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for f in range(start, end):
        ok, frame = cap.read()
        if not ok:
            break
        rel = f - center_frame
        sign = "p" if rel > 0 else ("m" if rel < 0 else "0")
        name = f"{label}_step{center_frame:04d}_{sign}{abs(rel):02d}.png"
        cv2.imwrite(str(out_dir / name), frame)
        written += 1
    cap.release()
    return written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument(
        "--kills",
        nargs="+",
        type=int,
        default=[1, 2],
        help="1-indexed kill events to extract (default: first two)",
    )
    ap.add_argument("--before", type=int, default=8, help="frames before each kill")
    ap.add_argument("--after", type=int, default=24, help="frames after each kill")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    kills = load_kills(args.trace)
    print(f"Found {len(kills)} kill events in trace.")
    if not kills:
        return

    for k_idx in args.kills:
        if not 1 <= k_idx <= len(kills):
            print(f"  skip: kill #{k_idx} out of range")
            continue
        k = kills[k_idx - 1]
        label = f"kill{k_idx:02d}"
        print(
            f"  extracting {label}: step={k['step']}  score+={k['score_delta']}  "
            f"~{k['game_sec']:.2f}s  life_lost={k['lives_dropped']}"
        )
        n = extract_window(
            video_path=args.video,
            center_frame=k["step"],
            before=args.before,
            after=args.after,
            out_dir=args.out,
            label=label,
        )
        print(f"    wrote {n} frames → {args.out}")


if __name__ == "__main__":
    main()
