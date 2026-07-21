import cv2
import numpy as np

class IrisDetector:
    def __init__(self, debug=False):
        self.debug = debug

    def process_image(self, img):
        """
        Main pipeline to detect pupil and iris boundaries.
        Returns a dictionary with pupil and iris circle coordinates (x, y, r).
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 1. Detect Pupil (Inner Circle)
        # Pupils are usually the darkest regions
        # We apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        
        # Detect circles using Hough Transform
        # Param1 is the Canny edge detection high threshold
        # Param2 is the accumulator threshold for the circle centers at the detection stage
        pupil_circles = cv2.HoughCircles(
            blurred, 
            cv2.HOUGH_GRADIENT, 
            dp=1.2, 
            minDist=100, 
            param1=100, 
            param2=30, 
            minRadius=10, 
            maxRadius=150
        )

        pupil = None
        if pupil_circles is not None:
            pupil_circles = np.round(pupil_circles[0, :]).astype("int")
            # Usually the pupil is the circle with the darkest center. 
            # We sort by the intensity of the center pixel to find the darkest one.
            pupil_circles = sorted(pupil_circles, key=lambda c: int(gray[c[1], c[0]]) if c[1] < gray.shape[0] and c[0] < gray.shape[1] else 255)
            pupil = pupil_circles[0] # (x, y, r)
        else:
            return {"error": "Pupil not found"}

        # 2. Detect Iris (Outer Circle)
        # The iris shares roughly the same center as the pupil but has a larger radius.
        px, py, pr = pupil
        
        # We look for a circle around the pupil's center
        iris_circles = cv2.HoughCircles(
            blurred, 
            cv2.HOUGH_GRADIENT, 
            dp=1.2, 
            minDist=100, 
            param1=100, 
            param2=40, 
            minRadius=pr + 20, 
            maxRadius=pr * 5
        )

        iris = None
        if iris_circles is not None:
            iris_circles = np.round(iris_circles[0, :]).astype("int")
            # We want the circle whose center is closest to the pupil's center
            iris_circles = sorted(iris_circles, key=lambda c: np.sqrt((c[0] - px)**2 + (c[1] - py)**2))
            iris = iris_circles[0]
        else:
            # Fallback estimation if Hough fails: Iris is usually ~2.5x - 3x the pupil radius
            iris = (px, py, int(pr * 2.8))

        # Compile results
        result = {
            "pupil": {"x": int(pupil[0]), "y": int(pupil[1]), "r": int(pupil[2])},
            "iris": {"x": int(iris[0]), "y": int(iris[1]), "r": int(iris[2])},
            "status": "success"
        }

        if self.debug:
            self._draw_debug(img.copy(), result)

        return result

    def _draw_debug(self, img, result):
        """Draws the detected circles on the image and saves it for debugging"""
        if result["status"] == "success":
            p = result["pupil"]
            i = result["iris"]
            # Draw pupil (green)
            cv2.circle(img, (p["x"], p["y"]), p["r"], (0, 255, 0), 2)
            cv2.circle(img, (p["x"], p["y"]), 2, (0, 0, 255), 3) # center
            
            # Draw iris (blue)
            cv2.circle(img, (i["x"], i["y"]), i["r"], (255, 0, 0), 2)
            
            cv2.imwrite("debug_iris_detection.jpg", img)

    def extract_iris_ring(self, img, pupil, iris):
        """
        Creates a mask isolating just the iris ring (excluding pupil and sclera).
        """
        mask = np.zeros(img.shape[:2], dtype="uint8")
        # Draw the full iris
        cv2.circle(mask, (iris["x"], iris["y"]), iris["r"], 255, -1)
        # Subtract the pupil
        cv2.circle(mask, (pupil["x"], pupil["y"]), pupil["r"], 0, -1)
        
        iris_isolated = cv2.bitwise_and(img, img, mask=mask)
        return iris_isolated
