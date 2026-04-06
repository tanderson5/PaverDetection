import cv2 as cv


def draw_paver(image, cnt, centroid, unique_corner, orientation_angle, approx_pts):
    """
    Draw all visual overlays for a single detected paver onto `image` in-place.

    Args:
        image:             BGR image to draw on.
        cnt:               Raw contour points for the paver.
        centroid:          (x, y) centre of the paver.
        unique_corner:     (x, y) of the farthest polygon corner (tab reference).
        orientation_angle: Computed lantern alignment angle in degrees.
        approx_pts:        Approximated polygon vertices from approxPolyDP.
    """
    # Contour outline
    cv.drawContours(image, [cnt], -1, (0, 255, 0), 2)

    # Polygon vertices
    for pt in approx_pts:
        x, y = pt[0]
        cv.circle(image, (x, y), 3, (0, 255, 255), -1)

    # Line from centroid to unique corner
    cv.line(image, centroid, unique_corner, (0, 0, 0), 2)

    # Centroid dot
    cv.circle(image, centroid, 6, (255, 0, 0), -1)

    # Orientation angle label
    label = f"{orientation_angle:.1f} deg"
    cv.putText(
        image, label,
        (centroid[0] + 10, centroid[1] - 10),
        cv.FONT_HERSHEY_SIMPLEX,
        0.6, (0, 0, 0), 2,
    )

    
