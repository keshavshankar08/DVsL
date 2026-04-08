import cv2 as cv
import numpy as np
import os
import time
import torch
from torchvision import transforms
from collections import deque
from PIL import Image
from cnn_backend import dvs_cnn

class dvsl_backend:
    def __init__(self, base_dir="data"):
        self.base_dir = base_dir
        self.classes = [chr(i) for i in range(ord('a'), ord('z') + 1)] + ['none']
        self.is_recording = False
        self.last_save_time = 0.0
        self.fps_interval = 1.0 / 10.0
        self.current_target_class = 'a'
        self.img_initial_gray_resized = None
        self._setup_directories()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = dvs_cnn(len(self.classes)).to(self.device)
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])

        self.prediction_buffer = deque(maxlen=30)
        self.last_pred_time = 0.0
        self.pred_interval = 1.0 / 30.0

        self.load_model()

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

    def save_frame(self, frame):
        if not self.is_recording:
            return

        current_time = time.time()
        if current_time - self.last_save_time >= self.fps_interval:
            filename = f"{current_time:.4f}.png"
            filepath = os.path.join(self.base_dir, self.current_target_class, filename)
            cv.imwrite(filepath, frame)
            self.last_save_time = current_time

    def load_model(self, path="asl_model.pth"):
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
            self.model.eval()
            return True
        else:
            return False

    def predict_character(self, temp_contrast_frame):
        current_time = time.time()
        
        if current_time - self.last_pred_time < self.pred_interval:
            return self._get_majority_vote()

        self.last_pred_time = current_time

        img = Image.fromarray(temp_contrast_frame)
        input_tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_tensor)
            _, predicted = torch.max(outputs.data, 1)
            pred_class = self.classes[predicted.item()]

        self.prediction_buffer.append(pred_class)

        return self._get_majority_vote()
    
    def _get_majority_vote(self):
        if not self.prediction_buffer:
            return "-"
        return max(set(self.prediction_buffer), key=self.prediction_buffer.count)
    
    def clear_buffer(self):
        self.prediction_buffer.clear()