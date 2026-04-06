# --------------------------
# CONFIGURATION
# --------------------------

# Detection thresholds
MIN_AREA = 1000
SOLIDITY_THRESHOLD = 0.94

# Pose estimation
THETA_OFFSET = -132.71951467531812  # measured offset for lantern alignment
TAB_ANGLE_OFFSET = 135              # degrees ± from unique corner to find tabs
MAX_RAY_RADIUS = 500                # max pixel distance for tab ray search
POLYGON_APPROX_EPSILON = 0.01      # fraction of arc length for approxPolyDP

# HSV red color ranges (red wraps around 0/180 in HSV)
LOWER_RED1 = (0,   100,  50)
UPPER_RED1 = (10,  255, 255)
LOWER_RED2 = (160, 100,  50)
UPPER_RED2 = (180, 255, 255)

# Debug: set to True to save intermediate images to disk
DEBUG = False
