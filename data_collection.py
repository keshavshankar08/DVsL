import cv2 as cv
import numpy as np
import os
import time

class DVsL:
    def __init__(self, base_dir="data"):
        self.base_dir = base_dir
        self.classes = [chr(i) for i in range(ord('a'), ord('z') + 1)] + ['none']
        self.current_idx = 0
        self.is_recording = False
        self.last_save_time = 0.0
        self.fps_interval = 1.0 / 10.0
        self._setup_directories()

    def _setup_directories(self):
        for cls in self.classes:
            os.makedirs(os.path.join(self.base_dir, cls), exist_ok=True)

    def resize_frame(self, image_initial):
        h, w = image_initial.shape
        min_dim = min(h, w)
        start_x = (w // 2) - (min_dim // 2)
        start_y = (h // 2) - (min_dim // 2)
        square_img = image_initial[start_y:start_y+min_dim, start_x:start_x+min_dim]
        return cv.resize(square_img, (128, 128))
    
    def temporal_contrast(self, image_initial, image):
        threshold = 20
        temp_contrast_frame = np.full(image_initial.shape, 127, dtype=np.uint8)
        diff_frame = image.astype(np.float32) - image_initial.astype(np.float32)
        temp_contrast_frame[diff_frame > threshold] = 255
        temp_contrast_frame[diff_frame < -threshold] = 0
        return temp_contrast_frame
    
    def toggle_recording(self):
        if self.current_idx >= len(self.classes):
            return False

        self.is_recording = not self.is_recording
        if not self.is_recording:
            self.current_idx += 1
        return True

    def save_frame(self, frame):
        if not self.is_recording or self.current_idx >= len(self.classes):
            return

        current_time = time.time()
        if current_time - self.last_save_time >= self.fps_interval:
            cls_name = self.classes[self.current_idx]
            filename = f"{current_time:.4f}.png"
            filepath = os.path.join(self.base_dir, cls_name, filename)
            cv.imwrite(filepath, frame)
            self.last_save_time = current_time

    def get_overlay_text(self):
        if self.current_idx >= len(self.classes):
            return "Complete"
        state = "REC" if self.is_recording else "PAUSED"
        cls_name = self.classes[self.current_idx].upper()
        return f"Class: {cls_name} [{state}]"
    
    def print_distribution_stats(self):
        total = 0
        print("\nDataset Distribution:")
        for cls in self.classes:
            path = os.path.join(self.base_dir, cls)
            count = len([f for f in os.listdir(path) if f.endswith('.png')]) if os.path.exists(path) else 0
            total += count
            print(f"{cls.upper()}: {count}")
        print(f"Total: {total}\n")

    def trim_and_print_stats(self, limit=100):
        total = 0
        print(f"\nDataset Distribution (Limit: {limit}):")
        for cls in self.classes:
            path = os.path.join(self.base_dir, cls)
            if not os.path.exists(path):
                print(f"{cls.upper()}: 0")
                continue

            files = sorted([f for f in os.listdir(path) if f.endswith('.png')])
            
            if len(files) > limit:
                for f in files[:-limit]:
                    os.remove(os.path.join(path, f))
                count = limit
            else:
                count = len(files)

            total += count
            print(f"{cls.upper()}: {count}")
            
        print(f"Total: {total}\n")

if __name__ == "__main__":
    dvsl = DVsL()

    camera = cv.VideoCapture(0)
    ret, img_initial = camera.read()
    img_initial = cv.flip(img_initial, 1)
    img_initial_gray = cv.cvtColor(img_initial, cv.COLOR_BGR2GRAY)

    image_initial_gray_resized = dvsl.resize_frame(img_initial_gray)

    while True:
        ret, image = camera.read()
        if not ret:
            break

        image = cv.flip(image, 1)
        image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        image_gray_resized = dvsl.resize_frame(image_gray)

        temporal_contrast_frame = dvsl.temporal_contrast(image_initial_gray_resized, image_gray_resized)

        dvsl.save_frame(temporal_contrast_frame)

        display_frame = image_gray_resized.copy()
        cv.putText(display_frame, dvsl.get_overlay_text(), (5, 15), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        cv.imshow("Original", display_frame)
        cv.imshow('Temporal Contrast', temporal_contrast_frame)

        image_initial_gray_resized = image_gray_resized

        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            if not dvsl.toggle_recording():
                break

    camera.release()
    cv.destroyAllWindows()