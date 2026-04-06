import cv2 as cv
import numpy as np

from config import (
    MIN_AREA, SOLIDITY_THRESHOLD, THETA_OFFSET,
    TAB_ANGLE_OFFSET, MAX_RAY_RADIUS, POLYGON_APPROX_EPSILON,
    LOWER_RED1, UPPER_RED1, LOWER_RED2, UPPER_RED2,
    DEBUG,
)


# --------------------------
# PREPROCESSING
# --------------------------
def preprocess(frame):
    # Convert BGR frame to a binary mask isolating red pixels.
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    mask1 = cv.inRange(hsv, np.array(LOWER_RED1), np.array(UPPER_RED1))
    mask2 = cv.inRange(hsv, np.array(LOWER_RED2), np.array(UPPER_RED2))
    
    mask = cv.bitwise_or(mask1, mask2)

    if DEBUG:
        cv.imwrite("debug_hsv_mask.jpg", mask)

    return mask


# --------------------------
# PAVER DETECTION
# --------------------------
def detect_pavers(mask):
    # Return contours of valid pavers from the binary mask.
    # Filters by minimum area and solidity to reject noise / partial shapes.
    
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    valid_pavers = []

    for cnt in contours:
        area = cv.contourArea(cnt)
        if area < MIN_AREA:
            continue

        hull = cv.convexHull(cnt)
        hull_area = cv.contourArea(hull)
        if hull_area == 0:
            continue

        solidity = area / hull_area
        if solidity < SOLIDITY_THRESHOLD:
            continue

        valid_pavers.append(cnt)

    return valid_pavers


# --------------------------
# POSE ESTIMATION
# --------------------------
def estimate_pose(cnt):
    # Estimate the pose (centroid + orientation) of a single paver contour.
    # Returns a dict with:
    #     centroid          - (x, y) pixel position
    #     unique_corner     - farthest polygon vertex from centroid (tab reference)
    #     orientation_angle - angle in degrees relative to THETA_OFFSET
    #     approx_pts        - polygon approximation vertices
    #     rect              - minAreaRect result
    
    rect = cv.minAreaRect(cnt)
    (cx, cy), _, _ = rect
    centroid = (int(cx), int(cy))

    # Approximate polygon and find the farthest corner as the tab reference
    epsilon = POLYGON_APPROX_EPSILON * cv.arcLength(cnt, True)
    approx = cv.approxPolyDP(cnt, epsilon, True)
    unique_corner = farthestCorner(approx, centroid)

    if unique_corner is None:
        return None

    dx = unique_corner[0] - centroid[0]
    dy = unique_corner[1] - centroid[1]
    theta = np.degrees(np.arctan2(dy, dx))
    orientation_angle = theta - THETA_OFFSET

    if DEBUG:
        import cv2 as cv2_dbg
        dbg = np.zeros((10, 10, 3), dtype=np.uint8)  # placeholder — caller draws
        print(f"[DEBUG] centroid={centroid}, theta={theta:.2f}, orient={orientation_angle:.2f}")

    return {
        "centroid": centroid,
        "unique_corner": unique_corner,
        "orientation_angle": orientation_angle,
        "approx_pts": approx,
        "rect": rect,
    }


def farthestCorner(approx, centroid):
    # Return the polygon vertex farthest from the centroid.
    best_corner = None
    max_dist = -1

    for pt in approx:
        x, y = pt[0]
        dist = np.hypot(x - centroid[0], y - centroid[1])
        if dist > max_dist:
            max_dist = dist
            best_corner = (x, y)

    return best_corner


# --------------------------
# TAB DETECTION VIA RAYS
# --------------------------
def find_tab_along_ray(mask, centroid, angle_deg, max_radius=MAX_RAY_RADIUS):
    # Walk outward from centroid along angle_deg until the mask goes dark.
    # Returns the first zero-mask pixel (the tab edge), or centroid as fallback.
    
    angle = np.radians(angle_deg)
    cx, cy = centroid

    for r in range(1, max_radius):
        x = int(cx + r * np.cos(angle))
        y = int(cy + r * np.sin(angle))

        if x < 0 or x >= mask.shape[1] or y < 0 or y >= mask.shape[0]:
            break
        if mask[y, x] == 0: # reached edge
            return (x, y)

    return centroid  # fallback if ray never exits mask


# --------------------------
# FRAME PIPELINE
# --------------------------
def process_frame(frame):
    # Run the full detection + pose pipeline on a single BGR frame.

    # Returns:
    #     pavers  - list of pose dicts (one per detected paver)
    #     mask    - binary red mask (useful for debug display)
    
    mask = preprocess(frame)
    contours = detect_pavers(mask)

    pavers = []
    for cnt in contours:
        pose = estimate_pose(cnt)
        if pose is None:
            continue

        # Attach the raw contour so callers can draw it
        pose["contour"] = cnt
        pavers.append(pose)

    return pavers, mask
