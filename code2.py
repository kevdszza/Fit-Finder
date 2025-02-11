import argparse
import cv2
import numpy as np
import math

# Define global variables and constants
refPt = []
r1 = 5  # for affine correction
r2 = 2  # for measurement
# Checkerboard measurements
ref_ht = 2.84
rectangle_row = 9
rectangle_col = 6
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
metre_pixel_x = 2
metre_pixel_y = 2
window_name1 = "image"
draw_radius = 10

def squ_point(img, x, y, k):
    time_pass = 50
    for i in range(time_pass):
        for j in range(time_pass):
            img[y - 25 + i, x - 25 + j] = np.array([10 * k, 50 * k, 0])

def click_and_crop(event, x, y, flags, param):
    global refPt
    if event == cv2.EVENT_LBUTTONUP:
        refPt.append((x, y))

def get_distance(image):
    global refPt
    refPt = []
    while True:
        cv2.imshow(window_name1, image)
        if len(refPt) == 2:
            break
        k = cv2.waitKey(1) & 0xFF
    cv2.destroyAllWindows()
    if len(refPt) == 2:
        pixel_dist_y = abs(refPt[0][1] - refPt[1][1])
        pixel_dist_x = abs(refPt[0][0] - refPt[1][0])
        actual_y = metre_pixel_y * pixel_dist_y
        actual_x = metre_pixel_x * pixel_dist_x
        actual_dist = math.sqrt(actual_y ** 2 + actual_x ** 2)
        return actual_dist
    return 0

def get_points(img):
    points = []
    img_to_show = img.copy()
    def draw_circle(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(img_to_show, (x, y), draw_radius, (255, 0, 0), -1)
            points.append([x, y])
    cv2.namedWindow(window_name1, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name1, img.shape[0], img.shape[1])
    cv2.setMouseCallback(window_name1, draw_circle)
    while True:
        cv2.imshow(window_name1, img_to_show)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()
    return points

def get_real_world_distance(points, m_x, m_y):
    pixel_dist_y = abs(points[0][1] - points[1][1])
    pixel_dist_x = abs(points[0][0] - points[1][0])
    actual_y = m_y * pixel_dist_y
    actual_x = m_x * pixel_dist_x
    actual_dist = math.sqrt(actual_y ** 2 + actual_x ** 2)
    return actual_dist

def get_waist(img, m_x, m_y):
    points = get_points(img)
    print("Selected points for waist measurement:", points)  # Debugging output
    if len(points) == 2:
        actual_dist = get_real_world_distance(points, m_x, m_y)
        print("Calculated waist distance (in pixels):", get_real_world_distance(points, 1, 1))  # Debugging output
        return actual_dist
    print("Insufficient points selected for waist measurement")  # Debugging output
    return 0

def chess_board_corners(image, gray, r):
    square_size = int(r + 1)
    ret, corners = cv2.findChessboardCorners(image, (rectangle_row, rectangle_col), None)
    if not ret:
        raise ValueError("Checkerboard pattern not detected.")
    cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    corners2 = corners
    coordinates = []
    coordinates.append((corners2[0, 0, 0], corners2[0, 0, 1]))
    coordinates.append((corners2[square_size - 1, 0, 0], corners2[square_size - 1, 0, 1]))
    coordinates.append((corners2[rectangle_row * (square_size - 1), 0, 0], corners2[rectangle_row * (square_size - 1), 0, 1]))
    coordinates.append((corners2[rectangle_row * (square_size - 1) + square_size - 1, 0, 0], corners2[rectangle_row * (square_size - 1) + square_size - 1, 0, 1]))
    return coordinates

def affine_correct_params(image):
    gray = np.copy(image)
    if len(image.shape) > 2:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    refPt = chess_board_corners(image, gray, r1)
    pt1 = np.asarray(refPt, dtype=np.float32)
    dist = (refPt[1][0] - refPt[0][0])
    refPt[1] = (refPt[0][0] + dist, refPt[0][1])
    refPt[2] = (refPt[0][0], refPt[0][1] + dist)
    refPt[3] = (refPt[0][0] + dist, refPt[0][1] + dist)
    pt2 = np.asarray(refPt, dtype=np.float32)
    M = cv2.getPerspectiveTransform(pt1, pt2)
    return M

def affine_correct(image, M=None):
    if M is None:
        M = affine_correct_params(image)
    image2 = np.copy(image)
    if len(image2) < 3:
        image2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2RGB)
    dst = cv2.warpPerspective(image2, M, (image.shape[1], image.shape[0]))
    return dst

def grub_cut(img, refPt):
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (refPt[0][0], refPt[0][1], refPt[1][0], refPt[1][1])
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 10, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img = img * mask2[:, :, np.newaxis]
    return img

def drawCircle(img, pt, state):
    img = img.astype(np.uint8)
    img_col = np.copy(img)
    if len(img_col.shape) < 3:
        img_col = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    cv2.circle(img_col, (pt[0], pt[1]), 10, (255, 0, 255), -1)
    if state == 0:
        while True:
            cv2.imshow('img', img_col)
            k = cv2.waitKey(20) & 0xFF
            if k == 27:
                break
        cv2.destroyAllWindows()
        return img
    else:
        return cv2.cvtColor(img_col, cv2.COLOR_BGR2GRAY)

def getHeadPoint(mask):
    shape = mask.shape
    y_head = (np.nonzero(np.sum(mask, axis=1)))[0][0]
    x_head = np.argmax(mask[y_head])
    return (x_head, y_head)

def first_sharp_fall(mask, x, y, win_size, thres):
    x_curr = x
    y0 = np.nonzero(mask[:, x_curr])[0][0]
    y0_diff = 10000
    x_curr = x + 1 * win_size
    y_curr = y0
    while True:
        if len(np.nonzero(mask[:, x_curr])[0]) == 0:
            x_curr = x_curr - 1 * win_size
            break
        y_curr = np.nonzero(mask[:, x_curr])[0][0]
        y_diff = y_curr - y0
        if y0_diff != 0:
            if float(y_diff) / float(y0_diff) > thres:
                break
        x_curr = x_curr + 1 * win_size
        y0_diff = y_diff
        y0 = y_curr
        if x_curr <= 0 or x_curr >= mask.shape[1]:
            break
    return (x_curr, y_curr)

def get_wrist(mask):
    thres = 20 * 255
    wrist_x_left = np.nonzero(np.sum(mask, axis=0) > thres)[0][0]
    wrist_y_left = np.argmax(mask[:, wrist_x_left])
    circled = drawCircle(mask, (wrist_x_left, wrist_y_left), draw_radius)
    nonzero = len(np.nonzero(np.sum(mask, axis=0) > thres)[0])
    wrist_x_right = np.nonzero(np.sum(mask, axis=0) > thres)[0][nonzero - 1]
    wrist_y_right = np.argmax(mask[:, wrist_x_right])
    circled = drawCircle(circled, (wrist_x_right, wrist_y_right), draw_radius)
    return ((wrist_x_left, wrist_y_left), (wrist_x_right, wrist_y_right))

def get_shoulder_to_waist(img, m_x, m_y):
    points = get_points(img)
    print("Selected points for shoulder to waist measurement:", points)  # Debugging output
    if len(points) == 2:
        shoulder_to_waist = get_real_world_distance(points, m_x, m_y)
        print("Calculated shoulder to waist distance (in pixels):", get_real_world_distance(points, 1, 1))  # Debugging output
        return shoulder_to_waist
    print("Insufficient points selected for shoulder to waist measurement")  # Debugging output
    return 0

def get_shoulder_to_knee(img, m_x, m_y):
    points = get_points(img)
    print("Selected points for shoulder to knee measurement:", points)  # Debugging output
    if len(points) == 2:
        shoulder_to_knee = get_real_world_distance(points, m_x, m_y)
        print("Calculated shoulder to knee distance (in pixels):", get_real_world_distance(points, 1, 1))  # Debugging output
        return shoulder_to_knee
    print("Insufficient points selected for shoulder to knee measurement")  # Debugging output
    return 0

def get_shoulder_length(img, m_x, m_y):
    points = get_points(img)
    print("Selected points for shoulder length measurement:", points)  # Debugging output
    if len(points) == 2:
        shoulder_length = get_real_world_distance(points, m_x, m_y)
        print("Calculated shoulder length distance (in pixels):", get_real_world_distance(points, 1, 1))  # Debugging output
        return shoulder_length
    print("Insufficient points selected for shoulder length measurement")  # Debugging output
    return 0

def get_chest(img, m_x, m_y):
    points = get_points(img)
    print("Selected points for chest measurement:", points)  # Debugging output
    if len(points) == 2:
        chest = get_real_world_distance(points, m_x, m_y)
        print("Calculated chest distance (in pixels):", get_real_world_distance(points, 1, 1))  # Debugging output
        return chest
    print("Insufficient points selected for chest measurement")  # Debugging output
    return 0

def get_sleeve(img, m_x, m_y):
    points = get_points(img)
    print("Selected points for sleeve measurement:", points)  # Debugging output
    if len(points) == 2:
        sleeve_length = get_real_world_distance(points, m_x, m_y)
        print("Calculated sleeve length distance (in pixels):", get_real_world_distance(points, 1, 1))  # Debugging output
        return sleeve_length
    print("Insufficient points selected for sleeve measurement")  # Debugging output
    return 0

def get_body_measurements(image):
    global metre_pixel_x, metre_pixel_y

    # Ensure we have valid metre_pixel_x and metre_pixel_y values
    print(f"Metre per pixel (x): {metre_pixel_x}, (y): {metre_pixel_y}")  # Debugging output
    
    image = affine_correct(image)
    waist = get_waist(image, metre_pixel_x, metre_pixel_y)
    chest = get_chest(image, metre_pixel_x, metre_pixel_y)
    sleeve_length = get_sleeve(image, metre_pixel_x, metre_pixel_y)
    shoulder_to_waist = get_shoulder_to_waist(image, metre_pixel_x, metre_pixel_y)
    shoulder_to_knee = get_shoulder_to_knee(image, metre_pixel_x, metre_pixel_y)
    shoulder_length = get_shoulder_length(image, metre_pixel_x, metre_pixel_y)
    
    # Convert measurements from meters to inches
    meter_to_inches = 39.3701
    waist_inch = waist * meter_to_inches
    chest_inch = chest * meter_to_inches
    sleeve_length_inch = sleeve_length * meter_to_inches
    shoulder_to_waist_inch = shoulder_to_waist * meter_to_inches
    shoulder_to_knee_inch = shoulder_to_knee * meter_to_inches
    shoulder_length_inch = shoulder_length * meter_to_inches

    print(f"Waist measurement: {waist_inch:.2f} inches")
    print(f"Chest measurement: {chest_inch:.2f} inches")
    print(f"Sleeve length measurement: {sleeve_length_inch:.2f} inches")
    print(f"Shoulder to waist length measurement: {shoulder_to_waist_inch:.2f} inches")
    print(f"Shoulder to knee length measurement: {shoulder_to_knee_inch:.2f} inches")
    print(f"Shoulder length measurement: {shoulder_length_inch:.2f} inches")

def main():
    ap = argparse.ArgumentParser(description="Measurement Tool")
    ap.add_argument("-i", "--image", required=True, help="Path to the input image")
    ap.add_argument("-s", "--scale", required=False, default=1, type=int, help="Scale of the image")
    args = vars(ap.parse_args())

    image_path = args["image"]
    scale = args["scale"]

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        return

    if scale != 1:
        image = cv2.resize(image, (int(image.shape[1] / scale), int(image.shape[0] / scale)))

    # Set the meter per pixel values here (update with actual values)
    global metre_pixel_x, metre_pixel_y
    metre_pixel_x = 0.002  # Example value; you should update this based on your measurements
    metre_pixel_y = 0.002  # Example value; you should update this based on your measurements

    get_body_measurements(image)

if __name__ == "__main__":
    main()
# python code2.py --image an1ov.jpg