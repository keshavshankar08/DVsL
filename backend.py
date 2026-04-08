import os
import time
import cv2 as cv
import numpy as np
import torch
from torchvision import transforms
from collections import deque
from PIL import Image
from typing import Optional, List
from tcasl import TCASL
from model_registry import LOCAL_MODEL_REGISTRY

class TCASLBackend:
    def __init__(self, default_arch: str = "cnn_v1", base_dir: str = "data") -> None:
        """
        Initializes the development backend with a specific architecture.

        :param default_arch: The string identifier for the target architecture.
        :param base_dir: The root directory for saving collected image data.
        """
        self.base_dir: str = base_dir
        self.classes: List[str] = [chr(i) for i in range(ord('a'), ord('z') + 1)] + ['none']
        self.is_recording: bool = False
        self.last_save_time: float = 0.0
        self.fps_interval: float = 1.0 / 10.0
        self.current_target_class: str = 'a'
        self.img_initial_gray_resized: Optional[np.ndarray] = None
        self.current_arch: str = default_arch
        
        self.tcasl_engine = TCASL()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])

        self.prediction_buffer: deque = deque(maxlen=30)
        self.last_pred_time: float = 0.0
        self.pred_interval: float = 1.0 / 30.0

        self._setup_directories()
        self.set_architecture(self.current_arch)

    def _setup_directories(self) -> None:
        """Creates target directories for all classification classes."""
        for cls in self.classes:
            os.makedirs(os.path.join(self.base_dir, cls), exist_ok=True)

    def set_architecture(self, arch: str) -> bool:
        """
        Switches the active neural network architecture.

        :param arch: The string key of the architecture from the registry.
        :return: True if successful, False if the architecture is not found.
        """
        if arch not in LOCAL_MODEL_REGISTRY:
            return False
            
        self.current_arch = arch
        ModelClass = LOCAL_MODEL_REGISTRY[arch]
        self.model = ModelClass(len(self.classes)).to(self.device)
        self.load_model()
        return True

    def process_frame(self, frame_gray: np.ndarray) -> np.ndarray:
        """
        Resizes the frame using the core engine logic.

        :param frame_gray: Grayscale input frame.
        :return: Cropped and resized 128x128 frame.
        """
        return self.tcasl_engine.preprocess_frame(frame_gray)

    def temporal_contrast(self, image_initial: np.ndarray, image: np.ndarray) -> np.ndarray:
        """
        Computes the temporal contrast between two frames using the core engine.

        :param image_initial: The previous frame.
        :param image: The current frame.
        :return: A trinary temporal contrast frame.
        """
        return self.tcasl_engine.compute_temporal_contrast(image_initial, image)

    def save_frame(self, frame: np.ndarray) -> None:
        """
        Saves the current frame to disk if recording is active.

        :param frame: The processed frame to save.
        """
        if not self.is_recording:
            return

        current_time = time.time()
        if current_time - self.last_save_time >= self.fps_interval:
            filename = f"{current_time:.4f}.png"
            filepath = os.path.join(self.base_dir, self.current_target_class, filename)
            cv.imwrite(filepath, frame)
            self.last_save_time = current_time

    def load_model(self) -> bool:
        """
        Loads local model weights into the currently active architecture.

        :return: True if successful, False otherwise.
        """
        path = f"{self.current_arch}.pth"
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
            self.model.eval()
            return True
        return False

    def predict_character(self, temp_contrast_frame: np.ndarray) -> str:
        """
        Runs local model inference on a processed frame.

        :param temp_contrast_frame: The temporal contrast input array.
        :return: The predicted string character.
        """
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
    
    def _get_majority_vote(self) -> str:
        """
        Calculates the most common prediction in the current buffer window.

        :return: The majority class string.
        """
        if not self.prediction_buffer:
            return "-"
        return max(set(self.prediction_buffer), key=self.prediction_buffer.count)
    
    def clear_buffer(self) -> None:
        """Empties the prediction queue."""
        self.prediction_buffer.clear()