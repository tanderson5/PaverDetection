"""
main.py - Entry point for paver detection.

Usage:
    # Single image
    python main.py --image path/to/image.jpg

    # Live webcam (default camera)
    python main.py --live

    # Live webcam with specific camera index
    python main.py --live --camera 1

    # Video file
    python main.py --video path/to/video.mp4
"""

import argparse

import cv2 as cv

from detection import process_frame
from visualization import draw_paver


# --------------------------
# IMAGE MODE
# --------------------------
def run_image(image_path):
    image = cv.imread(image_path)
    if image is None:
        print(f"[ERROR] Could not read image: {image_path}")
        return

    pavers, _ = process_frame(image)

    for pose in pavers:
        draw_paver(
            image,
            cnt=pose["contour"],
            centroid=pose["centroid"],
            unique_corner=pose["unique_corner"],
            orientation_angle=pose["orientation_angle"],
            approx_pts=pose["approx_pts"],
        )
        print(f"Paver @ {pose['centroid']}  angle={pose['orientation_angle']:.2f} deg")

    cv.imwrite("pose_detected.jpg", image)
    print(f"[INFO] Saved pose_detected.jpg  ({len(pavers)} paver(s) found)")

    cv.imshow("Paver Detection", image)
    cv.waitKey(0)
    cv.destroyAllWindows()


# --------------------------
# LIVE / VIDEO MODE
# --------------------------
def run_live(camera_index=0, video_path=None):
    source = video_path if video_path else camera_index
    cap = cv.VideoCapture(source)
    
    frame_count = 0

    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {source}")
        return

    print("[INFO] Press 'q' to quit, 's' to save current frame.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of stream.")
            break

        pavers, _ = process_frame(frame)

        for pose in pavers:
            draw_paver(
                frame,
                cnt=pose["contour"],
                centroid=pose["centroid"],
                unique_corner=pose["unique_corner"],
                orientation_angle=pose["orientation_angle"],
                approx_pts=pose["approx_pts"],
            )

        cv.namedWindow("Paver Detection (live)", cv.WINDOW_NORMAL)
        cv.setWindowProperty("Paver Detection (live)", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

        cv.imshow("Paver Detection (live)", frame)

        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"capture_{frame_count:04d}.jpg"
            cv.imwrite(filename, frame)
            print(f"[INFO] Saved {filename}")

        frame_count += 1

    cap.release()
    cv.destroyAllWindows()


# --------------------------
# ARGUMENT PARSING
# --------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Paver pose detection")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image",  type=str, help="Path to a single image file")
    group.add_argument("--live",   action="store_true", help="Use live webcam feed")
    group.add_argument("--video",  type=str, help="Path to a video file")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera index for --live mode (default: 0)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.image:
        run_image(args.image)
    elif args.live:
        run_live(camera_index=args.camera)
    elif args.video:
        run_live(video_path=args.video)
