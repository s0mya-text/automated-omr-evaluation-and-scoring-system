import cv2
import numpy as np
import argparse

def order_points(pts):
    """Orders the four corner points of the bubble grid."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] # Top-left
    rect[2] = pts[np.argmax(s)] # Bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # Top-right
    rect[3] = pts[np.argmax(diff)] # Bottom-left
    return rect

def perspective_transform(image, corners):
    """Applies a perspective transform to the image."""
    ordered_corners = order_points(corners)
    (tl, tr, br, bl) = ordered_corners
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0], [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(ordered_corners, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

# --- Main Script Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OMR Sheet detection using Hough Circles")
    parser.add_argument("-i", "--image", required=True, help="Path to the input image")
    args = vars(parser.parse_args())

    original_image = cv2.imread(args["image"])
    if original_image is None:
        print(f"Error: Could not load image at {args['image']}")
        exit()

    # 1. Pre-processing
    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    # Median blur is good for removing noise while preserving edges (and circles)
    blurred = cv2.medianBlur(gray, 5)

    # 2. Circle Detection using Hough Transform
    # This is the key step. It requires tuning based on the image resolution.
    # dp: Inverse ratio of accumulator resolution. 1 is same resolution.
    # minDist: Minimum distance between detected centers.
    # param1: Higher threshold for Canny edge detector (internal).
    # param2: Accumulator threshold for circle centers. Lower means more circles.
    # minRadius, maxRadius: The size of bubbles in your image.
    print("STEP 1: Detecting circles...")
    # NOTE: You may need to tune minRadius and maxRadius based on your image size
    # For a typical phone photo, bubble radius is around 5-15 pixels.
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
                               param1=50, param2=25, minRadius=5, maxRadius=15)

    if circles is None:
        print("Could not detect any circles. Try adjusting HoughCircles parameters.")
        exit()

    print(f"STEP 2: Found {len(circles[0])} circles. Identifying corners...")
    
    # 3. Find the corners of the grid formed by the circles
    points = circles[0][:, :2] # Extracting (x, y) coordinates of circle centers

    # Find the bounding box of all circle centers
    x, y, w, h = cv2.boundingRect(points.astype(np.int32))

    # The four corners of this bounding box are our new anchor points
    # Adding a small padding to ensure we capture the whole sheet
    padding = 10
    sheet_corners = np.array([
        [x - padding, y - padding],
        [x + w + padding, y - padding],
        [x + w + padding, y + h + padding],
        [x - padding, y + h + padding]
    ], dtype="float32")

    # 4. Draw the detected circles and the determined bounding box for debugging
    debug_image = original_image.copy()
    for (cx, cy, r) in circles[0]:
        cv2.circle(debug_image, (int(cx), int(cy)), int(r), (0, 255, 0), 2)
    
    # Draw the bounding rectangle
    cv2.rectangle(debug_image, (x, y), (x + w, y + h), (0, 0, 255), 3)
    cv2.imshow("Detected Circles and Bounding Box", cv2.resize(debug_image, (600, 800)))

    # 5. Apply the perspective transform
    rectified_image = perspective_transform(original_image, sheet_corners)
    print("STEP 3: Applied perspective transform.")

    output_filename = "rectified_image_circles.png"
    cv2.imwrite(output_filename, rectified_image)
    print(f"Successfully processed image. Rectified sheet saved as '{output_filename}'")
    
    cv2.imshow("Rectified Image", cv2.resize(rectified_image, (600, 800)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()