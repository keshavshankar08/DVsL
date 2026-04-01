import cv2 as cv
import numpy as np
import os
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import torch.nn as nn
from torchvision import transforms
from collections import deque
from PIL import Image
import re
import warnings
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
warnings.filterwarnings("ignore")
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()
hf_logging.disable_progress_bar()
from cnn_backend import dvs_cnn

class dvsl_backend:
    def __init__(self, base_dir="data", llm_name="Qwen/Qwen2.5-0.5B-Instruct"):
        self.base_dir = base_dir
        self.classes = [chr(i) for i in range(ord('a'), ord('z') + 1)] + ['none']
        self.is_recording = False
        self.last_save_time = 0.0
        self.fps_interval = 1.0 / 10.0
        self.current_target_class = 'a'
        self.img_initial_gray_resized = None
        self._setup_directories()
        self.llm = self._load_llm(llm_name)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = dvs_cnn(len(self.classes)).to(self.device)
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])

        self.prediction_buffer = deque(maxlen=15)
        self.last_pred_time = 0.0
        self.pred_interval = 1.0 / 10.0

        self.load_model()

    def _load_llm(self, model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            device_map="auto",
            torch_dtype=torch.float16
        )
        gen = pipeline("text-generation", model=model, tokenizer=tokenizer)
        return gen
    
    def _build_prompt(self, raw_sequence):
        messages = [
            {
                "role": "system", 
                "content": (
                    "You are a spelling corrector for a sign language interpreter. "
                    "The interpreter often repeats letters or swaps similar-looking letters "
                    "Simply respond with the corrected word they meant to spell."
                )
            },
            {"role": "user", "content": "Correct this: parson"},
            {"role": "assistant", "content": "person"},
            {"role": "user", "content": "Correct this: trucck"},
            {"role": "assistant", "content": "truck"},
            {"role": "user", "content": f"Correct this: {raw_sequence}"}
        ]
        return messages
    
    def _predict_word(self, messages):
        try:
            output = self.llm(
                messages, 
                max_new_tokens=8,
                max_length=None,
                temperature=None,
                top_p=None,
                top_k=None,
                do_sample=False,
                return_full_text=False
            )
            
            if not output or "generated_text" not in output[0]:
                return ""
            
            raw_output = output[0]["generated_text"]
            first_line = raw_output.split('\n')[0]
            clean_word = re.sub(r'<think>.*?</think>', '', first_line, flags=re.DOTALL)
            
            return clean_word.strip()
            
        except Exception as e:
            print(f"LLM Error: {e}")
            return ""

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

    def trim_and_print_stats(self, limit=100):
        total = 0
        for cls in self.classes:
            path = os.path.join(self.base_dir, cls)
            if not os.path.exists(path):
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
        print(f"Total samples: {total}")

    def clean_data(self):
        self.trim_and_print_stats(100)

    def train_model(self):
        return True

    def load_model(self, path="dvs_asl_model.pth"):
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

    def llm_correct(self, raw_sequence):
        #raw_sequence = "agpple" # for test
        prompt = self._build_prompt(raw_sequence)
        word = self._predict_word(prompt)
        return word