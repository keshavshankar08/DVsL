import os
import time
import numpy as np
import torch
from torchvision import transforms
from collections import deque
from PIL import Image
from typing import List, Tuple
from tcasl import TCASL
from model_registry import MODEL_REGISTRY

class TCASLBackend:
    def __init__(self, default_arch: str = "sdnn_v1", base_dir: str = "data") -> None:
        self.base_dir = base_dir
        self.classes = [chr(i) for i in range(ord('a'), ord('z') + 1)]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # State Management
        self.is_recording = False
        self.last_save_time = 0.0
        self.current_target_class = 'a'
        self.img_initial = None
        self.current_arch = default_arch
        
        # Prediction State
        self.prediction_buffer = deque(maxlen=30)
        self.last_pred_time = 0.0
        self.last_top5 = []

        # Core Engine for preprocessing & temporal contrast
        self.tcasl_engine = TCASL() 
        
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        # Setup and Load
        for cls in self.classes:
            os.makedirs(os.path.join(self.base_dir, cls), exist_ok=True)
        self.set_architecture(self.current_arch)

    def set_architecture(self, arch: str) -> bool:
        if arch not in MODEL_REGISTRY:
            return False
        self.current_arch = arch
        self.model = MODEL_REGISTRY[arch]["class"](len(self.classes)).to(self.device)        
        self.load_model()
        return True

    def load_model(self) -> bool:
        path = f"{self.current_arch}.pth"
        if not os.path.exists(path):
            print(f"Warning: {path} not found.")
            return False
    
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        # Streamlined SDNN buffer patch
        if "sdnn" in self.current_arch:
            for name, module in self.model.named_modules():
                for attr in ['running_mean', 'running_var']:
                    if hasattr(module, attr) and f"{name}.{attr}" in checkpoint:
                        target_shape = checkpoint[f"{name}.{attr}"].shape
                        module.register_buffer(attr, torch.zeros(target_shape).to(self.device))

        try:
            self.model.load_state_dict(checkpoint, strict=True)
            self.model.eval()
            print(f"Successfully loaded {self.current_arch} weights.")
            return True
        except RuntimeError as e:
            print(f"Strict load failed: {e}")
            return False

    def predict_character(self, temp_contrast_frame: np.ndarray) -> Tuple[str, List]:
        current_time = time.time()
        
        # Throttle predictions to 30 FPS
        if current_time - self.last_pred_time < (1.0 / 30.0):
            return self._get_majority_vote(), self.last_top5

        self.last_pred_time = current_time
        img = Image.fromarray(temp_contrast_frame)
        input_tensor = self.transform(img).unsqueeze(0).to(self.device)

        if "sdnn" in self.current_arch:
            input_tensor = input_tensor.unsqueeze(-1)

        with torch.no_grad():
            outputs = self.model(input_tensor)
            
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            logits = logits.flatten(start_dim=1) 

        probs = torch.softmax(logits, dim=1)[0]
        top_probs, top_indices = torch.topk(probs, 5)
        
        self.last_top5 = [(self.classes[idx.item()], prob.item() * 100) for prob, idx in zip(top_probs, top_indices)]
        self.prediction_buffer.append(self.classes[torch.argmax(logits, 1).item()])

        return self._get_majority_vote(), self.last_top5
    
    def _get_majority_vote(self) -> str:
        return max(set(self.prediction_buffer), key=self.prediction_buffer.count) if self.prediction_buffer else "-"
    
    def clear_buffer(self) -> None:
        self.prediction_buffer.clear()