import cv2 as cv
import numpy as np
import os
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import re
import warnings
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
warnings.filterwarnings("ignore")
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()
hf_logging.disable_progress_bar()

class dvsl_backend:
    def __init__(self, base_dir="data", llm_name="Qwen/Qwen3-0.6B"): # or Qwen/Qwen2.5-0.5B
        self.base_dir = base_dir
        self.classes = [chr(i) for i in range(ord('a'), ord('z') + 1)] + ['none']
        self.is_recording = False
        self.last_save_time = 0.0
        self.fps_interval = 1.0 / 10.0
        self.current_target_class = 'a'
        self.img_initial_gray_resized = None
        self._setup_directories()
        self.llm = self._load_llm(llm_name)

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
        prompt = (
            "Fix the spelling:\n"
            "helo -> hello\n"
            "wrold -> world\n"
            "bannana -> banana\n"
            "datta -> data\n"
            f"{raw_sequence} -> "
        )
        return prompt
    
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

    def load_model(self):
        return True

    def predict_character(self, temp_contrast_frame):
        return "a"

    def llm_correct(self, raw_sequence):
        #raw_sequence = "agpple" # for test
        prompt = self._build_prompt(raw_sequence)
        word = self._predict_word(prompt)
        return word