import cv2
import numpy as np
import os

class RetinalPipeline:
    def __init__(self):
        print("👁️ Initializing Retinal Preprocessing Engine...")

    def stage_1_geometry_crop(self, img):
        """
        Stage 1: Identifies the circular retinal perimeter mask 
        and crops out empty peripheral black padding without stretching.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Threshold to create a binary mask of the eye circle
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest bounding contour (the eye bulb)
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            # Crop the matrix to the bounding rectangle box
            return img[y:y+h, x:x+w]
        return img

    def stage_3_quality_gate(self, img):
        """
        Stage 3: Multi-property quality assessment tracking focus, 
        lighting, and contrast metrics using matrix variance.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        contrast = np.std(gray)
        # Laplacian variance measures image sharpness (blur detection)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Operational screening threshold boundaries
        is_usable = brightness > 15 and contrast > 8 and sharpness > 25
        metrics = {
            "Brightness Mean": round(brightness, 2),
            "Contrast StdDev": round(contrast, 2),
            "Sharpness Variance": round(sharpness, 2)
        }
        return is_usable, metrics

    def stage_4_optical_enhancement(self, img):
        """
        Stage 4: Applies CLAHE tile histogram equalization to bring out hidden 
        micro-pathologies and generates a green-spectrum Red-Free view.
        """
        # Convert to LAB color space to isolate illumination (L channel) from color info
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply Contrast Limited Adaptive Histogram Equalization
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        # Merge back and convert to standard color spectrum
        enhanced_lab = cv2.merge((cl, a, b))
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # Emulate clinical Red-Free Imaging by isolating the Green channel matrix
        _, g, _ = cv2.split(enhanced_bgr)
        red_free = cv2.merge([g, g, g])
        
        return enhanced_bgr, red_free

# --- Local Verification Routine ---
if __name__ == "__main__":
    # Create a dummy fundus-like test image matrix if no image is present
    dummy_img = np.zeros((500, 500, 3), dtype=np.uint8)
    cv2.circle(dummy_img, (250, 250), 200, (20, 45, 130), -1) # Red-orange fundus circle
    cv2.circle(dummy_img, (180, 250), 15, (40, 180, 240), -1) # Mock optic disc patch
    
    engine = RetinalPipeline()
    
    # Run Stage 1
    cropped = engine.stage_1_geometry_crop(dummy_img)
    print(f"✅ Stage 1 Complete. Matrix dimensions adjusted from {dummy_img.shape} to {cropped.shape}")
    
    # Run Stage 3
    passed, logs = engine.stage_3_quality_gate(cropped)
    print(f"📊 Stage 3 Quality Gate Status: Passed={passed} | Metrics: {logs}")
    
    # Run Stage 4
    enhanced, red_free_view = engine.stage_4_optical_enhancement(cropped)
    print("🎨 Stage 4 Complete. Local contrast matrices stabilized and Red-Free views extracted.")
