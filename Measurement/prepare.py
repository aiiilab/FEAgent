"""
Stereo Data Preparation Tool
=============================
Prepares synchronized stereo video data for main.py calibration and measurement.

Functions:
  1. sync    - Audio-based video synchronization (cross-correlation)
  2. extract - Extract calibration frames from checkerboard video
  3. prepare - Full pipeline: sync + extract + generate config

Requirements:
  pip install numpy scipy
  FFmpeg must be installed and in PATH

Author: Fish Stereo Vision Project
"""

import os
import sys
import subprocess
import argparse
import json
import struct
import wave
import tempfile
import shutil
from pathlib import Path

import numpy as np
from scipy import signal


# ============================================================
# SECTION 1: AUDIO SYNC
# ============================================================

def check_ffmpeg():
    """Verify FFmpeg is available."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True, text=True
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def extract_audio(video_path, output_wav, sample_rate=16000, duration=None):
    """
    Extract audio from video as mono WAV using FFmpeg.
    
    Args:
        video_path: Input video file
        output_wav: Output WAV file path
        sample_rate: Audio sample rate (16kHz is enough for sync)
        duration: Only extract first N seconds (None = all)
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",  # no video
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", "1",  # mono
    ]
    if duration:
        cmd.extend(["-t", str(duration)])
    cmd.append(output_wav)

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ERROR] FFmpeg audio extraction failed:")
        print(result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)
        return False
    return True


def read_wav(wav_path):
    """Read WAV file and return (samples_array, sample_rate)."""
    with wave.open(wav_path, 'r') as wf:
        n_frames = wf.getnframes()
        sr = wf.getframerate()
        raw = wf.readframes(n_frames)
        samples = np.array(struct.unpack(f'<{n_frames}h', raw), dtype=np.float64)
        # Normalize
        samples = samples / (np.max(np.abs(samples)) + 1e-10)
    return samples, sr


def find_audio_offset(audio1, audio2, sr, max_offset_sec=30):
    """
    Find time offset between two audio signals using cross-correlation.
    
    Returns:
        offset_seconds: Positive means audio2 is ahead (audio1 starts later)
        confidence: Peak correlation value (0-1)
    """
    max_offset_samples = int(max_offset_sec * sr)

    # Use only first 60 seconds for efficiency
    max_samples = min(len(audio1), len(audio2), 60 * sr)
    a1 = audio1[:max_samples]
    a2 = audio2[:max_samples]

    # Cross-correlation
    correlation = signal.correlate(a1, a2, mode='full')
    center = len(a2) - 1

    # Only search within max_offset range
    search_start = max(0, center - max_offset_samples)
    search_end = min(len(correlation), center + max_offset_samples)
    search_region = correlation[search_start:search_end]

    peak_idx = np.argmax(np.abs(search_region))
    peak_val = np.abs(search_region[peak_idx])

    # Convert to time offset
    offset_samples = (search_start + peak_idx) - center
    offset_seconds = offset_samples / sr

    # Confidence: ratio of peak to mean
    mean_val = np.mean(np.abs(search_region))
    confidence = peak_val / (mean_val + 1e-10)
    # Normalize confidence to 0-1 range (roughly)
    confidence = min(confidence / 20.0, 1.0)

    return offset_seconds, confidence


def sync_videos_multi(left_video, right_video, output_dir, max_offset=30, side_video=None):
    """
    Synchronize 2 or 3 videos using audio cross-correlation.
    Left video is the reference. All videos are aligned to a common start
    and trimmed to the shortest duration.

    Args:
        left_video: Path to left camera video (reference)
        right_video: Path to right camera video
        output_dir: Directory to save synced videos
        max_offset: Maximum expected offset in seconds
        side_video: Optional path to side camera video

    Returns:
        dict with sync info
    """
    print("=" * 50)
    print("VIDEO SYNCHRONIZATION")
    print("=" * 50)

    if not check_ffmpeg():
        print("[ERROR] FFmpeg not found. Install FFmpeg and add to PATH.")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    videos = {"left": left_video, "right": right_video}
    if side_video:
        videos["side"] = side_video
    
    print(f"Videos to sync: {len(videos)}")
    for name, path in videos.items():
        print(f"  {name}: {path}")

    # Step 1: Extract audio from all videos
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_files = {}
        audio_data = {}

        for name, path in videos.items():
            wav_path = os.path.join(tmpdir, f"{name}.wav")
            print(f"\nExtracting audio from {name} video...")
            if not extract_audio(path, wav_path, duration=120):
                sys.exit(1)
            samples, sr = read_wav(wav_path)
            wav_files[name] = wav_path
            audio_data[name] = samples
            print(f"  {name} audio: {len(samples)/sr:.2f}s")

        # Step 2: Find offsets relative to left (reference)
        offsets = {"left": 0.0}
        confidences = {"left": 1.0}

        for name in ["right", "side"]:
            if name not in audio_data:
                continue
            print(f"\nComputing offset: left vs {name}...")
            offset, conf = find_audio_offset(audio_data["left"], audio_data[name], sr, max_offset)
            offsets[name] = offset
            confidences[name] = conf
            print(f"  Offset: {offset:+.4f}s, Confidence: {conf:.3f}")
            if conf < 0.3:
                print(f"  [WARN] Low confidence for {name}!")

    # Step 3: Calculate trim points
    # offset > 0 means the other video starts EARLIER than left
    # offset < 0 means the other video starts LATER than left
    # 
    # To align: each video needs to be trimmed from its start by:
    #   trim_start[name] = max(0, -offset[name] + global_shift)
    # where global_shift ensures all trim_starts >= 0

    print(f"\n{'='*50}")
    print("SYNC RESULTS")
    print(f"{'='*50}")

    # Convert offsets to "start time relative to the earliest recording"
    # If left offset=0, right offset=+2.0 → right started 2s before left
    # So right needs 2s trimmed from start, left needs 0s
    #
    # If left offset=0, right offset=-1.5 → right started 1.5s after left
    # So left needs 1.5s trimmed from start, right needs 0s

    # absolute_start[name] = when this video started, relative to the earliest
    # offset > 0: other started earlier (negative absolute time)
    # offset < 0: other started later (positive absolute time)
    absolute_starts = {}
    absolute_starts["left"] = 0.0
    for name in ["right", "side"]:
        if name in offsets:
            absolute_starts[name] = -offsets[name]

    # Shift so the latest start = 0 (everyone trims to the latest starter)
    latest_start = max(absolute_starts.values())
    trim_starts = {name: latest_start - t for name, t in absolute_starts.items()}

    for name in trim_starts:
        direction = ""
        if trim_starts[name] > 0.01:
            direction = f"(trim {trim_starts[name]:.4f}s from head)"
        else:
            direction = "(no head trim needed)"
        print(f"  {name}: offset={offsets.get(name, 0):+.4f}s, confidence={confidences.get(name, 1):.3f} {direction}")

    # Step 4: Get durations and compute common duration
    print(f"\nGetting video durations...")
    durations = {}
    for name, path in videos.items():
        dur = get_video_duration(path)
        remaining = dur - trim_starts[name]
        durations[name] = remaining
        print(f"  {name}: total={dur:.2f}s, after head trim={remaining:.2f}s")

    common_duration = min(durations.values())
    print(f"\nCommon duration (shortest after trim): {common_duration:.2f}s")

    # Step 5: Trim all videos (head + tail)
    print(f"\nTrimming videos...")
    synced_paths = {}
    for name, path in videos.items():
        output_path = os.path.join(output_dir, f"{name}_synced.mp4")
        start = trim_starts[name]
        print(f"  {name}: start={start:.4f}s, duration={common_duration:.2f}s -> {output_path}")
        run_ffmpeg_trim_duration(path, output_path, start, common_duration)
        synced_paths[name] = output_path

    # Get final info
    fps = get_video_fps(synced_paths["left"])
    resolution = get_video_resolution(synced_paths["left"])

    sync_info = {
        "offsets": offsets,
        "confidences": confidences,
        "trim_starts": trim_starts,
        "common_duration": common_duration,
        "synced_videos": synced_paths,
        "fps": fps,
        "resolution": resolution,
        "originals": {name: os.path.abspath(path) for name, path in videos.items()},
    }

    print(f"\n{'='*50}")
    print("SYNCHRONIZATION COMPLETE")
    print(f"{'='*50}")
    for name, path in synced_paths.items():
        print(f"  {name}: {path}")
    print(f"  Duration: {common_duration:.2f}s")
    print(f"  FPS: {fps}, Resolution: {resolution}")
    print(f"  All videos now start at the same moment and have the same length.")

    return sync_info


def run_ffmpeg_trim_duration(input_path, output_path, start_seconds, duration):
    """Trim video: skip start_seconds from beginning, keep only duration seconds."""
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_seconds:.6f}",
        "-i", input_path,
        "-t", f"{duration:.6f}",
        "-c", "copy",
        "-avoid_negative_ts", "make_zero",
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ERROR] FFmpeg trim failed: {result.stderr[-300:]}")
        sys.exit(1)



def run_ffmpeg_trim(input_path, output_path, start_seconds):
    """Trim video starting from a given time point."""
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_seconds:.6f}",
        "-i", input_path,
        "-c", "copy",
        "-avoid_negative_ts", "make_zero",
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ERROR] FFmpeg trim failed: {result.stderr[-300:]}")
        sys.exit(1)


def run_ffmpeg_copy(input_path, output_path):
    """Copy video without re-encoding."""
    cmd = ["ffmpeg", "-y", "-i", input_path, "-c", "copy", output_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ERROR] FFmpeg copy failed: {result.stderr[-300:]}")
        sys.exit(1)


def get_video_fps(video_path):
    """Get video FPS using FFprobe."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "csv=p=0",
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0 and result.stdout.strip():
        fps_str = result.stdout.strip()
        if '/' in fps_str:
            num, den = fps_str.split('/')
            return round(int(num) / int(den), 2)
        return float(fps_str)
    return 24.0


def get_video_duration(video_path):
    """Get video duration in seconds."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "csv=p=0",
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0 and result.stdout.strip():
        return float(result.stdout.strip())
    return 0.0


def get_video_resolution(video_path):
    """Get video resolution as 'WxH' string."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0",
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip().replace(",", "x")
    return "unknown"


# ============================================================
# SECTION 2: FRAME EXTRACTION
# ============================================================

def extract_calibration_frames(left_video, right_video, output_dir, fps_extract=1):
    """
    Extract frames from checkerboard calibration videos.
    
    Args:
        left_video: Left camera checkerboard video
        right_video: Right camera checkerboard video
        output_dir: Output directory (will create left/ and right/ subfolders)
        fps_extract: Frames per second to extract (default: 1)
    """
    print("\n" + "=" * 50)
    print("FRAME EXTRACTION")
    print("=" * 50)

    left_dir = os.path.join(output_dir, "left")
    right_dir = os.path.join(output_dir, "right")
    os.makedirs(left_dir, exist_ok=True)
    os.makedirs(right_dir, exist_ok=True)

    print(f"Extracting at {fps_extract} fps...")

    # Left
    print(f"  Left video -> {left_dir}/")
    cmd_L = [
        "ffmpeg", "-y",
        "-i", left_video,
        "-vf", f"fps={fps_extract}",
        os.path.join(left_dir, "frame_%04d.png")
    ]
    result = subprocess.run(cmd_L, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ERROR] Left extraction failed: {result.stderr[-300:]}")
        sys.exit(1)

    # Right
    print(f"  Right video -> {right_dir}/")
    cmd_R = [
        "ffmpeg", "-y",
        "-i", right_video,
        "-vf", f"fps={fps_extract}",
        os.path.join(right_dir, "frame_%04d.png")
    ]
    result = subprocess.run(cmd_R, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ERROR] Right extraction failed: {result.stderr[-300:]}")
        sys.exit(1)

    # Count extracted frames
    left_count = len([f for f in os.listdir(left_dir) if f.endswith('.png')])
    right_count = len([f for f in os.listdir(right_dir) if f.endswith('.png')])

    print(f"\nExtracted:")
    print(f"  Left:  {left_count} frames")
    print(f"  Right: {right_count} frames")

    if left_count != right_count:
        # Trim to same count
        min_count = min(left_count, right_count)
        print(f"  [WARN] Frame count mismatch. Using first {min_count} from each.")
        trim_frames(left_dir, min_count)
        trim_frames(right_dir, min_count)

    return {"left_dir": left_dir, "right_dir": right_dir, "frame_count": min(left_count, right_count)}


def trim_frames(directory, keep_count):
    """Remove excess frames beyond keep_count."""
    files = sorted([f for f in os.listdir(directory) if f.endswith('.png')])
    for f in files[keep_count:]:
        os.remove(os.path.join(directory, f))


# ============================================================
# SECTION 3: CONFIG GENERATION
# ============================================================

def generate_config(output_dir, sync_info=None, extract_info=None, 
                    board_cols=9, board_rows=6, square_size=18.0,
                    left_fish_video=None, right_fish_video=None):
    """
    Generate a config JSON file with all parameters for main.py.
    
    Args:
        output_dir: Directory to save config
        sync_info: Sync results dict
        extract_info: Frame extraction results dict
        board_cols, board_rows: Checkerboard inner corners
        square_size: Square size in mm
        left_fish_video, right_fish_video: Fish recording videos (synced)
    """
    config = {
        "checkerboard": {
            "board_cols": board_cols,
            "board_rows": board_rows,
            "square_size_mm": square_size,
        },
        "calibration": {
            "calib_dir": extract_info["left_dir"].replace("/left", "").replace("\\left", "") if extract_info else None,
            "frame_count": extract_info["frame_count"] if extract_info else None,
        },
        "videos": {
            "left_fish_video": left_fish_video or (sync_info["synced_videos"].get("left") if sync_info else None),
            "right_fish_video": right_fish_video or (sync_info["synced_videos"].get("right") if sync_info else None),
            "side_fish_video": sync_info["synced_videos"].get("side") if sync_info else None,
        },
        "sync": {
            "offsets": sync_info["offsets"] if sync_info else None,
            "confidences": sync_info["confidences"] if sync_info else None,
            "common_duration": sync_info["common_duration"] if sync_info else None,
            "fps": sync_info["fps"] if sync_info else None,
            "resolution": sync_info["resolution"] if sync_info else None,
        } if sync_info else None,
    }

    config_path = os.path.join(output_dir, "stereo_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"\nConfig saved to: {config_path}")

    # Print usage hints
    print(f"\n{'='*50}")
    print("NEXT STEPS (copy-paste these commands):")
    print(f"{'='*50}")

    calib_dir = config["calibration"]["calib_dir"]
    sq = config["checkerboard"]["square_size_mm"]
    left_v = config["videos"]["left_fish_video"]
    right_v = config["videos"]["right_fish_video"]

    print(f"\n# Step 1: Calibrate")
    print(f"python main.py calibrate --calib_dir \"{calib_dir}\" --square_size {sq} --show")
    print(f"\n# Step 2: Verify rectification")
    if calib_dir:
        print(f"python main.py verify --calib_file calib_params.npz --left_img \"{calib_dir}/left/frame_0001.png\" --right_img \"{calib_dir}/right/frame_0001.png\"")
    print(f"\n# Step 3: Measure fish")
    if left_v and right_v:
        print(f"python main.py measure --left_video \"{left_v}\" --right_video \"{right_v}\" --calib_file calib_params.npz")

    return config


# ============================================================
# SECTION 4: MAIN CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Stereo Data Preparation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Sync two fish recording videos
  python prepare.py sync --left_video left.mp4 --right_video right.mp4

  # Extract calibration frames from checkerboard videos
  python prepare.py extract --left_video calib_left.mp4 --right_video calib_right.mp4

  # Full pipeline: sync fish videos + extract calib frames + generate config
  python prepare.py prepare --left_fish left_fish.mp4 --right_fish right_fish.mp4 --left_calib left_calib.mp4 --right_calib right_calib.mp4

  # Sync only (no calibration videos)
  python prepare.py sync --left_video left.mp4 --right_video right.mp4 --output_dir ./synced
        """
    )

    subparsers = parser.add_subparsers(dest="command")

    # --- Sync ---
    p_sync = subparsers.add_parser("sync", help="Synchronize two or three videos using audio")
    p_sync.add_argument("--left_video", required=True, help="Left camera video")
    p_sync.add_argument("--right_video", required=True, help="Right camera video")
    p_sync.add_argument("--side_video", default=None, help="Side camera video (optional)")
    p_sync.add_argument("--output_dir", default="./synced", help="Output directory (default: ./synced)")
    p_sync.add_argument("--max_offset", type=float, default=30, help="Max expected offset in seconds (default: 30)")

    # --- Extract ---
    p_ext = subparsers.add_parser("extract", help="Extract calibration frames from video")
    p_ext.add_argument("--left_video", required=True, help="Left checkerboard video")
    p_ext.add_argument("--right_video", required=True, help="Right checkerboard video")
    p_ext.add_argument("--output_dir", default="./calib_frames", help="Output directory (default: ./calib_frames)")
    p_ext.add_argument("--fps", type=float, default=1, help="Frames per second to extract (default: 1)")

    # --- Prepare (full pipeline) ---
    p_prep = subparsers.add_parser("prepare", help="Full pipeline: sync + extract + config")
    p_prep.add_argument("--left_fish", required=True, help="Left camera fish recording video")
    p_prep.add_argument("--right_fish", required=True, help="Right camera fish recording video")
    p_prep.add_argument("--side_fish", default=None, help="Side camera fish recording video (optional)")
    p_prep.add_argument("--left_calib", required=True, help="Left camera checkerboard video")
    p_prep.add_argument("--right_calib", required=True, help="Right camera checkerboard video")
    p_prep.add_argument("--output_dir", default="./workspace", help="Output directory (default: ./workspace)")
    p_prep.add_argument("--board_cols", type=int, default=9, help="Checkerboard inner corners cols (default: 9)")
    p_prep.add_argument("--board_rows", type=int, default=6, help="Checkerboard inner corners rows (default: 6)")
    p_prep.add_argument("--square_size", type=float, default=18.0, help="Square size in mm (default: 18.0)")
    p_prep.add_argument("--fps", type=float, default=1, help="Calibration frame extraction fps (default: 1)")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == "sync":
        sync_info = sync_videos_multi(
            args.left_video, args.right_video,
            args.output_dir, args.max_offset,
            side_video=args.side_video
        )
        # Save sync info
        info_path = os.path.join(args.output_dir, "sync_info.json")
        with open(info_path, "w") as f:
            json.dump(sync_info, f, indent=2)
        print(f"\nSync info saved to: {info_path}")

    elif args.command == "extract":
        extract_info = extract_calibration_frames(
            args.left_video, args.right_video,
            args.output_dir, args.fps
        )
        print(f"\nCalibration frames ready in: {args.output_dir}")
        print(f"  Run: python main.py calibrate --calib_dir \"{args.output_dir}\"")

    elif args.command == "prepare":
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Step 1: Sync fish videos
        print("\n[STEP 1/3] Synchronizing fish recording videos...\n")
        sync_dir = os.path.join(output_dir, "synced_fish")
        sync_info = sync_videos_multi(
            args.left_fish, args.right_fish,
            sync_dir,
            side_video=getattr(args, 'side_fish', None)
        )

        # Step 2: Sync + extract calibration frames
        print("\n[STEP 2/3] Synchronizing and extracting calibration frames...\n")

        # First sync calib videos
        calib_sync_dir = os.path.join(output_dir, "synced_calib")
        calib_sync_info = sync_videos_multi(
            args.left_calib, args.right_calib,
            calib_sync_dir
        )

        # Then extract frames from synced calib videos
        calib_frames_dir = os.path.join(output_dir, "calib_frames")
        extract_info = extract_calibration_frames(
            calib_sync_info["synced_videos"]["left"],
            calib_sync_info["synced_videos"]["right"],
            calib_frames_dir,
            args.fps
        )

        # Step 3: Generate config
        print("\n[STEP 3/3] Generating config...\n")
        config = generate_config(
            output_dir,
            sync_info=sync_info,
            extract_info=extract_info,
            board_cols=args.board_cols,
            board_rows=args.board_rows,
            square_size=args.square_size,
        )

        print(f"\n{'='*50}")
        print("DATA PREPARATION COMPLETE")
        print(f"{'='*50}")
        print(f"Workspace: {output_dir}/")
        print(f"  synced_fish/     - Synchronized fish videos")
        print(f"  synced_calib/    - Synchronized calibration videos")
        print(f"  calib_frames/    - Extracted calibration images")
        print(f"  stereo_config.json - All parameters")


if __name__ == "__main__":
    main()
