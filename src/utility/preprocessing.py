import cv2
import numpy as np
from PIL import Image
from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import canny
from skimage.color import rgb2gray

class ImagePreprocessor:
    @staticmethod
    def load_image(image_file):
        """Convert upload file/path to OpenCV format (BGR)"""
        if isinstance(image_file, str):
            image = Image.open(image_file)
        else:
            image = Image.open(image_file)
        
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    @staticmethod
    def order_points(pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    @staticmethod
    def four_point_transform(image, pts):
        rect = ImagePreprocessor.order_points(pts)
        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    @staticmethod
    def robust_receipt_scanner(img_array):
        orig = img_array.copy()
        ratio = img_array.shape[0] / 500.0
        h = 500
        w = int(img_array.shape[1] / ratio)
        img_small = cv2.resize(img_array, (w, h))
        
        hsv = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)
        saturation = hsv[:,:,1]
        
        _, binary = cv2.threshold(saturation, 40, 255, cv2.THRESH_BINARY_INV)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        
        cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
        
        screenCnt = None
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.04 * peri, True)
            if len(approx) == 4:
                screenCnt = approx
                break
                
        if screenCnt is None and len(cnts) > 0:
            
            rect = cv2.minAreaRect(cnts[0])
            box = cv2.boxPoints(rect)
            screenCnt = np.int0(box)

        if screenCnt is None:
            return orig 

        screenCnt = screenCnt.astype("float32") * ratio
        return ImagePreprocessor.four_point_transform(orig, screenCnt.reshape(4, 2))

    @staticmethod
    def get_mean_error(data):
        if not data: return 0, 0
        median = np.median(data)
        mean_error = np.mean([abs(t - median) for t in data])
        return median, mean_error

    @staticmethod
    def normalize_vertical_angle(angle_deg):
        return angle_deg - 180 if angle_deg > 90 else angle_deg

    @staticmethod
    def get_rotation_candidates(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.blur(gray, (4, 4))

        edges = canny(gray) 

        tested_angles = np.deg2rad(np.arange(0.1, 180.0))
        h, theta, d = hough_line(edges, theta=tested_angles)
        _, angles, dists = hough_line_peaks(h, theta, d)

        horizontal_lines = []
        vertical_lines = []

        for angle in angles:
            angle_deg = np.rad2deg(angle)
            if 75 <= angle_deg <= 105:
                horizontal_lines.append(angle_deg)
            elif angle_deg <= 15 or angle_deg >= 165:
                vertical_lines.append(angle_deg)

        v_median, v_mean_error = ImagePreprocessor.get_mean_error(
            [ImagePreprocessor.normalize_vertical_angle(t) for t in vertical_lines]
        )
        h_median, h_mean_error = ImagePreprocessor.get_mean_error(horizontal_lines)

        if v_mean_error < h_mean_error:
            return [90, -90]
        else:
            return [0, 180]

    @staticmethod
    def choose_rotation(img, candidates):
        best_rotation = candidates[0]
        best_score = -1
        
        for angle in candidates:
            if angle == 0: rotated = img
            elif angle == 90: rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            elif angle == -90: rotated = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            else: rotated = cv2.rotate(img, cv2.ROTATE_180)
            
            gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            h_projection = np.sum(binary, axis=1)
            h_variance = np.var(h_projection)
            h_mean = np.mean(h_projection)
            h_peaks = np.sum(h_projection > h_mean * 1.5)
            
            score = h_variance * 0.5 + h_peaks * 50
            
            if score > best_score:
                best_score = score
                best_rotation = angle
        
        return best_rotation

    @staticmethod
    def process_image(image_file) -> Image.Image:
        """Main Pipeline: Load -> Warp -> Detect Rotation -> Correct Rotation"""
        # 1. Load (BGR)
        img = ImagePreprocessor.load_image(image_file)
        
        # 2. Warp / Scan
        warped = ImagePreprocessor.robust_receipt_scanner(img)
        
        # 3. Detect Candidates (Hough)
        candidates = ImagePreprocessor.get_rotation_candidates(warped)
        
        # 4. Fix Orientation (Projection Score)
        best_angle = ImagePreprocessor.choose_rotation(warped, candidates)
        
        final_img = warped
        if best_angle == 90:
            final_img = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
        elif best_angle == -90:
            final_img = cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif best_angle == 180:
            final_img = cv2.rotate(warped, cv2.ROTATE_180)

        return Image.fromarray(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB))