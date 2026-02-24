"""
Stereo Vision Fish Length Measurement Tool
==========================================
A single-file tool for:
  1. Stereo camera calibration from checkerboard images
  2. Loading calibration parameters + manual input
  3. Interactive fish length measurement from synchronized video pairs

Usage:
  python stereo_fish_measure.py calibrate --calib_dir ./calibration_images
  python stereo_fish_measure.py measure --left_video left.mp4 --right_video right.mp4 --calib_file calib_params.npz

Requirements:
  pip install opencv-python numpy

Author: Fish Stereo Vision Project
"""

import cv2
import numpy as np
import os
import sys
import glob
import argparse
import json
from pathlib import Path


# ============================================================
# SECTION 1: STEREO CALIBRATION
# ============================================================

def find_checkerboard_corners(image_path, board_size, show=False):
    """Detect checkerboard corners in a single image."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"  [WARN] Cannot read: {image_path}")
        return None, None, None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    ret, corners = cv2.findChessboardCorners(gray, board_size, flags)

    if ret:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        if show:
            vis = img.copy()
            cv2.drawChessboardCorners(vis, board_size, corners, ret)
            cv2.imshow("Corners", cv2.resize(vis, (960, 640)))
            cv2.waitKey(500)
    return ret, corners, gray.shape[::-1]


def stereo_calibrate(calib_dir, board_size=(9, 6), square_size=30.0, show=False):
    """
    Perform stereo calibration from paired checkerboard images.

    Args:
        calib_dir: Directory containing 'left/' and 'right/' subfolders with images
        board_size: (cols, rows) inner corner count, e.g. (9, 6)
        square_size: Physical size of each square in mm
        show: Show detected corners

    Returns:
        Dictionary with all calibration parameters
    """
    left_dir = os.path.join(calib_dir, "left")
    right_dir = os.path.join(calib_dir, "right")

    if not os.path.isdir(left_dir) or not os.path.isdir(right_dir):
        print(f"[ERROR] Expected 'left/' and 'right/' subfolders in: {calib_dir}")
        sys.exit(1)

    # Find image pairs sorted by filename
    exts = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif"]
    left_images = []
    right_images = []
    for ext in exts:
        left_images.extend(glob.glob(os.path.join(left_dir, ext)))
        right_images.extend(glob.glob(os.path.join(right_dir, ext)))
    left_images = sorted(left_images)
    right_images = sorted(right_images)

    if len(left_images) != len(right_images):
        print(f"[ERROR] Mismatch: {len(left_images)} left images vs {len(right_images)} right images")
        sys.exit(1)
    if len(left_images) == 0:
        print(f"[ERROR] No images found in {left_dir} or {right_dir}")
        sys.exit(1)

    print(f"Found {len(left_images)} image pairs")
    print(f"Board: {board_size[0]}x{board_size[1]} inner corners, square = {square_size}mm")
    print()

    # Prepare 3D object points
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    objp *= square_size

    objpoints = []
    imgpoints_L = []
    imgpoints_R = []
    img_size = None
    used_pairs = []

    for i, (l_path, r_path) in enumerate(zip(left_images, right_images)):
        l_name = os.path.basename(l_path)
        r_name = os.path.basename(r_path)
        print(f"  [{i+1}/{len(left_images)}] {l_name} + {r_name} ... ", end="")

        ret_L, corners_L, size_L = find_checkerboard_corners(l_path, board_size, show)
        ret_R, corners_R, size_R = find_checkerboard_corners(r_path, board_size, show)

        if ret_L and ret_R:
            if img_size is None:
                img_size = size_L
            objpoints.append(objp)
            imgpoints_L.append(corners_L)
            imgpoints_R.append(corners_R)
            used_pairs.append((l_name, r_name))
            print("OK")
        else:
            reason = []
            if not ret_L:
                reason.append("left failed")
            if not ret_R:
                reason.append("right failed")
            print(f"SKIP ({', '.join(reason)})")

    if show:
        cv2.destroyAllWindows()

    if len(objpoints) < 5:
        print(f"\n[ERROR] Only {len(objpoints)} valid pairs. Need at least 5 for reliable calibration.")
        sys.exit(1)

    print(f"\nUsing {len(objpoints)} valid pairs for calibration...")

    # Step 1: Individual camera calibration
    print("  Calibrating left camera...")
    ret_L, K1, D1, rvecs_L, tvecs_L = cv2.calibrateCamera(
        objpoints, imgpoints_L, img_size, None, None
    )
    print(f"    RMS reprojection error: {ret_L:.4f} px")

    print("  Calibrating right camera...")
    ret_R, K2, D2, rvecs_R, tvecs_R = cv2.calibrateCamera(
        objpoints, imgpoints_R, img_size, None, None
    )
    print(f"    RMS reprojection error: {ret_R:.4f} px")

    # Step 2: Stereo calibration
    print("  Running stereo calibration...")
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
    flags = cv2.CALIB_FIX_INTRINSIC

    ret_stereo, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_L, imgpoints_R,
        K1, D1, K2, D2, img_size,
        criteria=criteria, flags=flags
    )
    print(f"    Stereo RMS reprojection error: {ret_stereo:.4f} px")

    # Step 3: Stereo rectification
    print("  Computing rectification transforms...")
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K1, D1, K2, D2, img_size, R, T, alpha=0
    )

    # Step 4: Compute undistort+rectify maps
    map1_L, map2_L = cv2.initUndistortRectifyMap(K1, D1, R1, P1, img_size, cv2.CV_32FC1)
    map1_R, map2_R = cv2.initUndistortRectifyMap(K2, D2, R2, P2, img_size, cv2.CV_32FC1)

    # Extract baseline
    baseline_mm = abs(T[0, 0])
    focal_px = P1[0, 0]

    print(f"\n{'='*50}")
    print(f"CALIBRATION RESULTS")
    print(f"{'='*50}")
    print(f"  Image size:        {img_size[0]} x {img_size[1]}")
    print(f"  Valid pairs used:  {len(objpoints)}")
    print(f"  Stereo RMS error:  {ret_stereo:.4f} px")
    print(f"  Baseline (T_x):    {baseline_mm:.2f} mm")
    print(f"  Focal length (px): {focal_px:.2f}")
    print(f"  Left K:\n{K1}")
    print(f"  Right K:\n{K2}")
    print(f"{'='*50}")

    if ret_stereo > 1.0:
        print("\n[WARN] Stereo RMS > 1.0 px. Calibration quality may be poor.")
        print("       Consider retaking calibration images with better coverage.")

    # Pack all parameters
    calib = {
        "K1": K1, "D1": D1, "K2": K2, "D2": D2,
        "R": R, "T": T, "E": E, "F": F,
        "R1": R1, "R2": R2, "P1": P1, "P2": P2, "Q": Q,
        "roi1": np.array(roi1), "roi2": np.array(roi2),
        "map1_L": map1_L, "map2_L": map2_L,
        "map1_R": map1_R, "map2_R": map2_R,
        "img_size": np.array(img_size),
        "board_size": np.array(board_size),
        "square_size": np.array(square_size),
        "stereo_rms": np.array(ret_stereo),
        "baseline_mm": np.array(baseline_mm),
        "focal_px": np.array(focal_px),
    }
    return calib


def save_calibration(calib, output_path):
    """Save calibration parameters to .npz file."""
    np.savez(output_path, **calib)
    print(f"\nCalibration saved to: {output_path}")


def load_calibration(calib_path):
    """Load calibration parameters from .npz file."""
    if not os.path.exists(calib_path):
        print(f"[ERROR] Calibration file not found: {calib_path}")
        sys.exit(1)
    data = np.load(calib_path)
    calib = {key: data[key] for key in data.files}
    print(f"Loaded calibration from: {calib_path}")
    print(f"  Image size:  {calib['img_size'][0]} x {calib['img_size'][1]}")
    print(f"  Baseline:    {float(calib['baseline_mm']):.2f} mm")
    print(f"  Focal (px):  {float(calib['focal_px']):.2f}")
    print(f"  Stereo RMS:  {float(calib['stereo_rms']):.4f} px")
    return calib


# ============================================================
# SECTION 2: STEREO POINT MATCHING
# ============================================================

def rectify_images(img_L, img_R, calib):
    """Apply stereo rectification to an image pair."""
    rect_L = cv2.remap(img_L, calib["map1_L"], calib["map2_L"], cv2.INTER_LINEAR)
    rect_R = cv2.remap(img_R, calib["map1_R"], calib["map2_R"], cv2.INTER_LINEAR)
    return rect_L, rect_R


def find_corresponding_point(rect_L, rect_R, point_L, patch_size=81, search_range=600):
    """
    Find the corresponding point in the right image using template matching.

    After rectification, corresponding points lie on the same horizontal line (epipolar constraint).
    We search along that row in the right image in BOTH directions.

    Args:
        rect_L: Rectified left image
        rect_R: Rectified right image
        point_L: (x, y) pixel coordinates in rectified left image
        patch_size: Size of the template patch (odd number)
        search_range: Maximum horizontal search range in pixels

    Returns:
        (x, y) in rectified right image, disparity value, confidence score
    """
    x_L, y_L = int(point_L[0]), int(point_L[1])
    half = patch_size // 2
    h, w = rect_L.shape[:2]

    # Extract template patch from left image
    y_top = max(0, y_L - half)
    y_bot = min(h, y_L + half + 1)
    x_left = max(0, x_L - half)
    x_right = min(w, x_L + half + 1)

    if len(rect_L.shape) == 3:
        template = cv2.cvtColor(rect_L[y_top:y_bot, x_left:x_right], cv2.COLOR_BGR2GRAY)
    else:
        template = rect_L[y_top:y_bot, x_left:x_right].copy()

    if template.size == 0:
        return None, None, 0.0

    # Search strip in right image: BOTH directions from the left image point
    search_x_start = max(0, x_L - search_range)
    search_x_end = min(w, x_L + search_range)

    strip_y_top = max(0, y_L - half)
    strip_y_bot = min(h, y_L + half + 1)

    if len(rect_R.shape) == 3:
        strip = cv2.cvtColor(rect_R[strip_y_top:strip_y_bot, search_x_start:search_x_end], cv2.COLOR_BGR2GRAY)
    else:
        strip = rect_R[strip_y_top:strip_y_bot, search_x_start:search_x_end].copy()

    if strip.shape[1] < template.shape[1] or strip.shape[0] < template.shape[0]:
        return None, None, 0.0

    # Template matching
    result = cv2.matchTemplate(strip, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    # Subpixel refinement via parabola fitting
    best_x_in_strip = max_loc[0]
    if 1 <= best_x_in_strip < result.shape[1] - 1:
        left_val = result[0, best_x_in_strip - 1]
        center_val = result[0, best_x_in_strip]
        right_val = result[0, best_x_in_strip + 1]
        denom = 2.0 * (2.0 * center_val - left_val - right_val)
        if abs(denom) > 1e-6:
            subpixel_offset = (left_val - right_val) / denom
            best_x_in_strip += subpixel_offset

    # Convert back to full image coordinates
    x_R = search_x_start + best_x_in_strip + half
    y_R = y_L  # same row after rectification

    disparity = x_L - x_R
    return (float(x_R), float(y_R)), float(disparity), float(max_val)


def compute_3d_point(x, y, disparity, calib):
    """
    Compute 3D coordinates from pixel + disparity using Q matrix.

    Returns (X, Y, Z) in mm.
    """
    Q = calib["Q"]
    point_4d = Q @ np.array([x, y, disparity, 1.0])
    point_3d = point_4d[:3] / point_4d[3]
    return point_3d


def compute_fish_length(point_L_mouth, point_L_tail, rect_L, rect_R, calib):
    """
    Compute 3D fish length from two points marked on the left image.

    Args:
        point_L_mouth: (x, y) mouth position in rectified left image
        point_L_tail:  (x, y) caudal base position in rectified left image
        rect_L, rect_R: Rectified image pair
        calib: Calibration parameters

    Returns:
        length_mm, mouth_3d, tail_3d, match_info
    """
    # Find corresponding points in right image
    pt_R_mouth, disp_mouth, conf_mouth = find_corresponding_point(
        rect_L, rect_R, point_L_mouth
    )
    pt_R_tail, disp_tail, conf_tail = find_corresponding_point(
        rect_L, rect_R, point_L_tail
    )

    if pt_R_mouth is None or pt_R_tail is None:
        return None, None, None, {"error": "Template matching failed"}

    # Compute 3D positions
    mouth_3d = compute_3d_point(point_L_mouth[0], point_L_mouth[1], disp_mouth, calib)
    tail_3d = compute_3d_point(point_L_tail[0], point_L_tail[1], disp_tail, calib)

    # Euclidean distance
    length_mm = float(np.linalg.norm(mouth_3d - tail_3d))

    match_info = {
        "mouth_L": point_L_mouth,
        "mouth_R": pt_R_mouth,
        "tail_L": point_L_tail,
        "tail_R": pt_R_tail,
        "disparity_mouth": disp_mouth,
        "disparity_tail": disp_tail,
        "confidence_mouth": conf_mouth,
        "confidence_tail": conf_tail,
        "depth_mouth_mm": float(mouth_3d[2]),
        "depth_tail_mm": float(tail_3d[2]),
        "mouth_3d": mouth_3d.tolist(),
        "tail_3d": tail_3d.tolist(),
    }
    return length_mm, mouth_3d, tail_3d, match_info


# ============================================================
# SECTION 3: INTERACTIVE VIDEO MEASUREMENT
# ============================================================

class FishMeasurementApp:
    """Interactive GUI for measuring fish length from stereo video."""

    def __init__(self, left_video, right_video, calib):
        self.cap_L = cv2.VideoCapture(left_video)
        self.cap_R = cv2.VideoCapture(right_video)
        self.calib = calib

        if not self.cap_L.isOpened():
            print(f"[ERROR] Cannot open left video: {left_video}")
            sys.exit(1)
        if not self.cap_R.isOpened():
            print(f"[ERROR] Cannot open right video: {right_video}")
            sys.exit(1)

        self.total_frames = int(self.cap_L.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap_L.get(cv2.CAP_PROP_FPS)
        self.frame_idx = 0

        # Display scaling (4K is too large for screen)
        self.orig_w = int(self.cap_L.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.orig_h = int(self.cap_L.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.display_w = min(1280, self.orig_w)
        self.scale = self.display_w / self.orig_w
        self.display_h = int(self.orig_h * self.scale)

        # Clicked points (in ORIGINAL resolution)
        # Manual mode: 4 points total
        # [0] = left mouth, [1] = left tail, [2] = right mouth, [3] = right tail
        self.points_L = []  # [(x, y), (x, y)]  mouth, then tail on LEFT
        self.points_R = []  # [(x, y), (x, y)]  mouth, then tail on RIGHT
        self.click_phase = "left"  # "left" or "right"
        self.measurements = []

        # Current frames
        self.frame_L = None
        self.frame_R = None
        self.rect_L = None
        self.rect_R = None

        print(f"\nVideo: {self.orig_w}x{self.orig_h} @ {self.fps:.1f}fps, {self.total_frames} frames")
        print(f"Display scale: {self.scale:.2f}x ({self.display_w}x{self.display_h})")

    def read_frame(self, idx):
        """Read a specific frame from both videos."""
        self.cap_L.set(cv2.CAP_PROP_POS_FRAMES, idx)
        self.cap_R.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret_L, frame_L = self.cap_L.read()
        ret_R, frame_R = self.cap_R.read()
        if not ret_L or not ret_R:
            return False
        self.frame_L = frame_L
        self.frame_R = frame_R
        self.rect_L, self.rect_R = rectify_images(frame_L, frame_R, self.calib)
        self.frame_idx = idx
        return True

    def mouse_callback_L(self, event, x, y, flags, param):
        """Handle mouse clicks on the left image."""
        if event == cv2.EVENT_LBUTTONDOWN and self.click_phase == "left" and len(self.points_L) < 2:
            orig_x = x / self.scale
            orig_y = y / self.scale
            self.points_L.append((orig_x, orig_y))
            label = "Mouth" if len(self.points_L) == 1 else "Caudal base"
            print(f"  LEFT  {label}: ({orig_x:.1f}, {orig_y:.1f})")

            if len(self.points_L) == 2:
                self.click_phase = "right"
                print(f"\n  >> Now click the SAME fish in the RIGHT window <<")
                print(f"  >> Click mouth first, then caudal base <<")

    def mouse_callback_R(self, event, x, y, flags, param):
        """Handle mouse clicks on the right image."""
        if event == cv2.EVENT_LBUTTONDOWN and self.click_phase == "right" and len(self.points_R) < 2:
            orig_x = x / self.scale
            orig_y = y / self.scale
            self.points_R.append((orig_x, orig_y))
            label = "Mouth" if len(self.points_R) == 1 else "Caudal base"
            print(f"  RIGHT {label}: ({orig_x:.1f}, {orig_y:.1f})")

            if len(self.points_R) == 2:
                self.do_measurement()

    def do_measurement(self):
        """Perform measurement with the four manually clicked points."""
        pt_L_mouth = self.points_L[0]
        pt_L_tail = self.points_L[1]
        pt_R_mouth = self.points_R[0]
        pt_R_tail = self.points_R[1]

        # Compute disparities
        disp_mouth = pt_L_mouth[0] - pt_R_mouth[0]
        disp_tail = pt_L_tail[0] - pt_R_tail[0]

        # Compute 3D points
        Q = self.calib["Q"]
        mouth_4d = Q @ np.array([pt_L_mouth[0], pt_L_mouth[1], disp_mouth, 1.0])
        mouth_3d = mouth_4d[:3] / mouth_4d[3]

        tail_4d = Q @ np.array([pt_L_tail[0], pt_L_tail[1], disp_tail, 1.0])
        tail_3d = tail_4d[:3] / tail_4d[3]

        length_mm = float(np.linalg.norm(mouth_3d - tail_3d))

        print(f"\n  {'='*40}")
        print(f"  FISH LENGTH: {length_mm:.1f} mm")
        print(f"  {'='*40}")
        print(f"  Mouth depth:  {mouth_3d[2]:.1f} mm")
        print(f"  Tail depth:   {tail_3d[2]:.1f} mm")
        print(f"  Disparity:    mouth={disp_mouth:.2f}px, tail={disp_tail:.2f}px")
        print(f"  Mouth 3D:     ({mouth_3d[0]:.1f}, {mouth_3d[1]:.1f}, {mouth_3d[2]:.1f})")
        print(f"  Tail 3D:      ({tail_3d[0]:.1f}, {tail_3d[1]:.1f}, {tail_3d[2]:.1f})")

        # Sanity checks
        if disp_mouth <= 0 or disp_tail <= 0:
            print(f"  [WARN] Negative disparity! Left/right videos may be swapped.")
        if abs(mouth_3d[2] - tail_3d[2]) > 100:
            print(f"  [WARN] Large depth difference between mouth and tail ({abs(mouth_3d[2] - tail_3d[2]):.0f}mm)")

        result = {
            "frame": self.frame_idx,
            "length_mm": length_mm,
            "mouth_L": list(pt_L_mouth),
            "mouth_R": list(pt_R_mouth),
            "tail_L": list(pt_L_tail),
            "tail_R": list(pt_R_tail),
            "disparity_mouth": disp_mouth,
            "disparity_tail": disp_tail,
            "depth_mouth_mm": float(mouth_3d[2]),
            "depth_tail_mm": float(tail_3d[2]),
            "mouth_3d": mouth_3d.tolist(),
            "tail_3d": tail_3d.tolist(),
        }
        self.measurements.append(result)

    def draw_display_L(self):
        """Draw the left frame with annotations."""
        if self.rect_L is None:
            return np.zeros((self.display_h, self.display_w, 3), dtype=np.uint8)

        vis = self.rect_L.copy()

        # Draw clicked points
        colors = [(0, 0, 255), (0, 255, 0)]  # red=mouth, green=tail
        labels = ["Mouth", "Caudal"]
        for i, pt in enumerate(self.points_L):
            px = (int(pt[0]), int(pt[1]))
            cv2.circle(vis, px, 8, colors[i], 2)
            cv2.circle(vis, px, 2, colors[i], -1)
            cv2.putText(vis, labels[i], (px[0] + 12, px[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[i], 2)

        # Draw line between points
        if len(self.points_L) == 2:
            p1 = (int(self.points_L[0][0]), int(self.points_L[0][1]))
            p2 = (int(self.points_L[1][0]), int(self.points_L[1][1]))
            cv2.line(vis, p1, p2, (255, 255, 0), 2)
            if self.measurements:
                last = self.measurements[-1]
                mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2 - 15)
                text = f"{last['length_mm']:.1f} mm"
                cv2.putText(vis, text, mid, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)

        display = cv2.resize(vis, (self.display_w, self.display_h))

        # HUD
        active = ">> CLICK HERE <<" if self.click_phase == "left" else ""
        hud_lines = [
            f"LEFT IMAGE {active}",
            f"Frame: {self.frame_idx}/{self.total_frames-1}  |  Time: {self.frame_idx/self.fps:.2f}s",
            f"Keys: A/D=prev/next | W/S=jump10 | Q/E=jump100 | R=reset | Esc=quit",
        ]
        for i, line in enumerate(hud_lines):
            color = (0, 255, 255) if i == 0 and self.click_phase == "left" else (255, 255, 255)
            cv2.putText(display, line, (10, 25 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

        return display

    def draw_display_R(self):
        """Draw the right frame with annotations."""
        if self.rect_R is None:
            return np.zeros((self.display_h, self.display_w, 3), dtype=np.uint8)

        vis = self.rect_R.copy()

        # Draw clicked points
        colors = [(0, 0, 255), (0, 255, 0)]
        labels = ["Mouth", "Caudal"]
        for i, pt in enumerate(self.points_R):
            px = (int(pt[0]), int(pt[1]))
            cv2.circle(vis, px, 8, colors[i], 2)
            cv2.circle(vis, px, 2, colors[i], -1)
            cv2.putText(vis, labels[i], (px[0] + 12, px[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[i], 2)

        if len(self.points_R) == 2:
            p1 = (int(self.points_R[0][0]), int(self.points_R[0][1]))
            p2 = (int(self.points_R[1][0]), int(self.points_R[1][1]))
            cv2.line(vis, p1, p2, (255, 255, 0), 2)

        # Draw epipolar lines for left points (horizontal lines at same y)
        for pt in self.points_L:
            y_display = int(pt[1])
            cv2.line(vis, (0, y_display), (vis.shape[1], y_display), (0, 255, 255), 1)

        display = cv2.resize(vis, (self.display_w, self.display_h))

        active = ">> CLICK HERE <<" if self.click_phase == "right" else ""
        hud_lines = [
            f"RIGHT IMAGE {active}",
            f"Click the SAME fish: mouth then caudal base",
            f"Yellow lines = epipolar guides (fish should be on these lines)",
        ]
        for i, line in enumerate(hud_lines):
            color = (0, 255, 255) if i == 0 and self.click_phase == "right" else (255, 255, 255)
            cv2.putText(display, line, (10, 25 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

        return display

    def run(self):
        """Main interactive loop."""
        print("\n" + "=" * 50)
        print("MANUAL STEREO MEASUREMENT MODE")
        print("=" * 50)
        print("1. Navigate to a frame with a side-view fish")
        print("2. In LEFT window: click MOUTH, then CAUDAL BASE")
        print("3. In RIGHT window: click the SAME fish's MOUTH, then CAUDAL BASE")
        print("   (Use yellow epipolar lines as guides)")
        print("4. Read the measured length")
        print("5. Press R to reset, navigate to next fish")
        print("=" * 50)

        cv2.namedWindow("Left (click here first)", cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("Right (click same fish)", cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback("Left (click here first)", self.mouse_callback_L)
        cv2.setMouseCallback("Right (click same fish)", self.mouse_callback_R)

        self.read_frame(0)

        while True:
            display_L = self.draw_display_L()
            display_R = self.draw_display_R()
            cv2.imshow("Left (click here first)", display_L)
            cv2.imshow("Right (click same fish)", display_R)

            key = cv2.waitKey(30) & 0xFF

            if key == 27:  # Esc
                break
            elif key == ord('d'):  # next frame
                if self.frame_idx < self.total_frames - 1:
                    self.read_frame(self.frame_idx + 1)
                    self.reset_points()
            elif key == ord('a'):  # prev frame
                if self.frame_idx > 0:
                    self.read_frame(self.frame_idx - 1)
                    self.reset_points()
            elif key == ord('s'):  # jump forward 10
                new_idx = min(self.frame_idx + 10, self.total_frames - 1)
                self.read_frame(new_idx)
                self.reset_points()
            elif key == ord('w'):  # jump back 10
                new_idx = max(self.frame_idx - 10, 0)
                self.read_frame(new_idx)
                self.reset_points()
            elif key == ord('e'):  # jump forward 100
                new_idx = min(self.frame_idx + 100, self.total_frames - 1)
                self.read_frame(new_idx)
                self.reset_points()
            elif key == ord('q'):  # jump back 100
                new_idx = max(self.frame_idx - 100, 0)
                self.read_frame(new_idx)
                self.reset_points()
            elif key == ord('r'):  # reset points
                self.reset_points()
                print("  Points reset")

        cv2.destroyAllWindows()
        self.cap_L.release()
        self.cap_R.release()

        return self.measurements

    def reset_points(self):
        """Reset all clicked points."""
        self.points_L = []
        self.points_R = []
        self.click_phase = "left"

    def save_results(self, output_path):
        """Save all measurements to JSON."""
        if not self.measurements:
            print("No measurements to save.")
            return

        # Convert numpy types to python types for JSON
        def convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, tuple):
                return list(obj)
            return obj

        clean = []
        for m in self.measurements:
            clean.append({k: convert(v) for k, v in m.items()})

        with open(output_path, "w") as f:
            json.dump(clean, f, indent=2)
        print(f"\n{len(clean)} measurements saved to: {output_path}")


# ============================================================
# SECTION 4: VERIFICATION UTILITIES
# ============================================================

def verify_rectification(calib, left_img_path, right_img_path):
    """Show rectified images side by side with horizontal lines to verify alignment."""
    img_L = cv2.imread(left_img_path)
    img_R = cv2.imread(right_img_path)
    if img_L is None or img_R is None:
        print("[ERROR] Cannot read images for verification")
        return

    rect_L, rect_R = rectify_images(img_L, img_R, calib)

    # Stack horizontally
    combined = np.hstack([rect_L, rect_R])

    # Draw horizontal lines every 50 pixels
    h = combined.shape[0]
    for y in range(0, h, 50):
        cv2.line(combined, (0, y), (combined.shape[1], y), (0, 255, 0), 1)

    # Resize for display
    scale = 1280.0 / combined.shape[1]
    display = cv2.resize(combined, (1280, int(combined.shape[0] * scale)))

    print("\nRectification verification:")
    print("  Green lines should pass through the same features in both images.")
    print("  Press any key to close.")

    cv2.imshow("Rectification Verification (left | right)", display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ============================================================
# SECTION 5: MAIN CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Stereo Vision Fish Length Measurement Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Step 1: Calibrate from checkerboard images
  python stereo_fish_measure.py calibrate --calib_dir ./calibration_images

  # Step 1b: Calibrate with custom board size
  python stereo_fish_measure.py calibrate --calib_dir ./calibration_images --board_cols 9 --board_rows 6 --square_size 30

  # Step 2: Verify rectification quality
  python stereo_fish_measure.py verify --calib_file calib_params.npz --left_img calib/left/01.png --right_img calib/right/01.png

  # Step 3: Measure fish interactively
  python stereo_fish_measure.py measure --left_video left.mp4 --right_video right.mp4 --calib_file calib_params.npz
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # --- Calibrate ---
    p_cal = subparsers.add_parser("calibrate", help="Calibrate stereo cameras from checkerboard images")
    p_cal.add_argument("--calib_dir", required=True, help="Directory with left/ and right/ subfolders")
    p_cal.add_argument("--board_cols", type=int, default=9, help="Inner corners horizontally (default: 9)")
    p_cal.add_argument("--board_rows", type=int, default=6, help="Inner corners vertically (default: 6)")
    p_cal.add_argument("--square_size", type=float, default=30.0, help="Square size in mm (default: 30.0)")
    p_cal.add_argument("--output", default="calib_params.npz", help="Output calibration file")
    p_cal.add_argument("--show", action="store_true", help="Show detected corners")

    # --- Verify ---
    p_ver = subparsers.add_parser("verify", help="Verify rectification quality")
    p_ver.add_argument("--calib_file", required=True, help="Calibration .npz file")
    p_ver.add_argument("--left_img", required=True, help="Left image path")
    p_ver.add_argument("--right_img", required=True, help="Right image path")

    # --- Measure ---
    p_mea = subparsers.add_parser("measure", help="Interactive fish length measurement")
    p_mea.add_argument("--left_video", required=True, help="Left camera video file")
    p_mea.add_argument("--right_video", required=True, help="Right camera video file")
    p_mea.add_argument("--calib_file", required=True, help="Calibration .npz file")
    p_mea.add_argument("--output", default="measurements.json", help="Output measurements file")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == "calibrate":
        print("=" * 50)
        print("STEREO CALIBRATION")
        print("=" * 50)

        # Allow manual override
        print(f"\nBoard size: {args.board_cols} x {args.board_rows} inner corners")
        print(f"Square size: {args.square_size} mm")
        user_input = input("Press Enter to confirm, or type 'cols rows square_mm' to change: ").strip()
        if user_input:
            parts = user_input.split()
            if len(parts) == 3:
                args.board_cols = int(parts[0])
                args.board_rows = int(parts[1])
                args.square_size = float(parts[2])
                print(f"Updated: {args.board_cols}x{args.board_rows}, square={args.square_size}mm")

        calib = stereo_calibrate(
            args.calib_dir,
            board_size=(args.board_cols, args.board_rows),
            square_size=args.square_size,
            show=args.show
        )
        save_calibration(calib, args.output)

    elif args.command == "verify":
        calib = load_calibration(args.calib_file)
        verify_rectification(calib, args.left_img, args.right_img)

    elif args.command == "measure":
        print("=" * 50)
        print("STEREO FISH MEASUREMENT")
        print("=" * 50)
        calib = load_calibration(args.calib_file)

        app = FishMeasurementApp(args.left_video, args.right_video, calib)
        measurements = app.run()
        app.save_results(args.output)

        if measurements:
            print(f"\n{'='*50}")
            print("MEASUREMENT SUMMARY")
            print(f"{'='*50}")
            for i, m in enumerate(measurements):
                conf_note = ""
                if m["confidence_mouth"] < 0.5 or m["confidence_tail"] < 0.5:
                    conf_note = " [LOW CONFIDENCE]"
                print(f"  Fish {i+1}: {m['length_mm']:.1f} mm  "
                      f"(frame {m['frame']}, depth ~{m['depth_mouth_mm']:.0f}mm){conf_note}")


if __name__ == "__main__":
    main()
