import cv2 as cv
import tkinter as tk
import time
import os
from tkinter import ttk, messagebox, simpledialog
from PIL import Image, ImageTk

from backend import TCASLBackend
from model_registry import MODEL_REGISTRY

class TCASLFrontend:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("TCASL Dev Environment")
        self.backend = TCASLBackend()
        
        self.camera = None
        self.stream_running = False
        self.is_testing = False
        self.top5_widgets = []
        
        self._setup_ui()

    def _setup_ui(self) -> None:
        # Top Controls
        top_frame = ttk.Frame(self.root, padding=10)
        top_frame.pack(fill=tk.X)

        ttk.Label(top_frame, text="Camera:").pack(side=tk.LEFT)
        self.camera_var = tk.StringVar(value="0")
        ttk.Combobox(top_frame, textvariable=self.camera_var, values=["0", "1", "2", "3"], width=5).pack(side=tk.LEFT, padx=5)

        self.btn_stream = ttk.Button(top_frame, text="Start Stream", command=self.toggle_stream)
        self.btn_stream.pack(side=tk.LEFT, padx=5)
        ttk.Button(top_frame, text="Admin Access", command=self.request_admin).pack(side=tk.RIGHT)

        # Video Displays
        video_frame = ttk.Frame(self.root, padding=10)
        video_frame.pack()
        self.lbl_orig = ttk.Label(video_frame)
        self.lbl_orig.pack(side=tk.LEFT, padx=10)
        self.lbl_tc = ttk.Label(video_frame)
        self.lbl_tc.pack(side=tk.LEFT, padx=10)

        # Main Containers
        self.user_frame = ttk.Frame(self.root, padding=10)
        self.admin_frame = ttk.Frame(self.root, padding=10)
        
        self._build_user_view()
        self._build_admin_view()
        self.user_frame.pack(fill=tk.BOTH, expand=True)

    def _build_user_view(self) -> None:
        ttk.Label(self.user_frame, text="Realtime ASL Classifier", font=('', 14, 'bold')).pack(pady=5)
        self.btn_test = ttk.Button(self.user_frame, text="Start Predicting", command=self.toggle_testing)
        self.btn_test.pack(pady=5)
        
        ttk.Label(self.user_frame, text="Current Sign (Majority Vote):").pack()
        self.lbl_current_pred = ttk.Label(self.user_frame, text="-", font=('', 64, 'bold'), foreground="blue")
        self.lbl_current_pred.pack(pady=10)

        top5_frame = ttk.LabelFrame(self.user_frame, text="Top 5 Confidence", padding=10)
        top5_frame.pack(fill=tk.X, padx=50, pady=5)
        
        for _ in range(5):
            frame = ttk.Frame(top5_frame)
            frame.pack(fill=tk.X, pady=2)
            lbl = ttk.Label(frame, text="-: 0.0%", width=10)
            lbl.pack(side=tk.LEFT)
            bar = ttk.Progressbar(frame, length=200, mode='determinate', maximum=100)
            bar.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self.top5_widgets.append((lbl, bar))

    def _build_admin_view(self) -> None:
        ttk.Label(self.admin_frame, text="Admin Control Panel", font=('', 14, 'bold')).pack(pady=5)
        ttk.Button(self.admin_frame, text="Exit Admin", command=lambda: self._switch_view(self.user_frame)).pack(pady=5)

        # Data Collection
        data_frame = ttk.LabelFrame(self.admin_frame, text="Data Collection", padding=10)
        data_frame.pack(fill=tk.X, pady=5)
        ttk.Label(data_frame, text="Class:").pack(side=tk.LEFT)
        self.class_var = tk.StringVar(value="a")
        ttk.Combobox(data_frame, textvariable=self.class_var, values=self.backend.classes, width=5).pack(side=tk.LEFT, padx=5)
        self.btn_record = ttk.Button(data_frame, text="Start Recording", command=self.toggle_recording)
        self.btn_record.pack(side=tk.LEFT, padx=5)

        # Architecture Swapping
        train_frame = ttk.LabelFrame(self.admin_frame, text="Model Pipeline", padding=10)
        train_frame.pack(fill=tk.X, pady=5)
        ttk.Label(train_frame, text="Architecture:").pack(side=tk.LEFT)
        self.arch_var = tk.StringVar(value=self.backend.current_arch)
        ttk.Combobox(train_frame, textvariable=self.arch_var, values=list(MODEL_REGISTRY.keys()), width=10).pack(side=tk.LEFT, padx=5)
        ttk.Button(train_frame, text="Load & Switch Model", command=self.switch_and_load).pack(side=tk.LEFT, padx=5)

    def _switch_view(self, target_frame: ttk.Frame) -> None:
        self.user_frame.pack_forget()
        self.admin_frame.pack_forget()
        target_frame.pack(fill=tk.BOTH, expand=True)

    def request_admin(self) -> None:
        if simpledialog.askstring("Admin Login", "Enter Password:", show='*') == "admin":
            self._switch_view(self.admin_frame)
        else:
            messagebox.showerror("Error", "Incorrect Password")

    def switch_and_load(self) -> None:
        arch = self.arch_var.get()
        if self.backend.set_architecture(arch):
            msg = "Success" if self.backend.load_model() else "Architecture loaded, but weights missing."
            messagebox.showinfo("Status", f"{msg}")
        else:
            messagebox.showerror("Error", "Invalid architecture.")

    def toggle_stream(self) -> None:
        self.stream_running = not self.stream_running
        if self.stream_running:
            self.camera = cv.VideoCapture(int(self.camera_var.get()))
            if self.camera.isOpened():
                self.btn_stream.config(text="Stop Stream")
                self.process_video()
            else:
                self.stream_running = False
                messagebox.showerror("Error", "Could not open camera")
        else:
            self.btn_stream.config(text="Start Stream")
            if self.camera: self.camera.release()
            self.backend.img_initial = None

    def toggle_recording(self) -> None:
        self.backend.is_recording = not self.backend.is_recording
        self.backend.current_target_class = self.class_var.get()
        self.btn_record.config(text="Stop Recording" if self.backend.is_recording else "Start Recording")

    def toggle_testing(self) -> None:
        self.is_testing = not self.is_testing
        self.btn_test.config(text="Stop Predicting" if self.is_testing else "Start Predicting")
        self.backend.clear_buffer()
        self.lbl_current_pred.config(text="-")
        if not self.is_testing:
            for lbl, bar in self.top5_widgets:
                lbl.config(text="-: 0.0%")
                bar['value'] = 0

    def process_video(self) -> None:
        if not self.stream_running or not self.camera: return

        ret, frame = self.camera.read()
        if ret:
            frame_gray_resized = self.backend.tcasl_engine.preprocess_frame(cv.cvtColor(cv.flip(frame, 1), cv.COLOR_BGR2GRAY))
            
            if self.backend.img_initial is None:
                self.backend.img_initial = frame_gray_resized

            tc_frame = self.backend.tcasl_engine.compute_temporal_contrast(self.backend.img_initial, frame_gray_resized)

            if self.backend.is_recording:
                current_time = time.time()
                if current_time - self.backend.last_save_time >= (1.0 / 10.0):
                    cv.imwrite(os.path.join(self.backend.base_dir, self.backend.current_target_class, f"{current_time:.4f}.png"), tc_frame)
                    self.backend.last_save_time = current_time

            if self.is_testing:
                pred, top5 = self.backend.predict_character(tc_frame)
                self.lbl_current_pred.config(text=pred.upper())
                for i, (char, prob) in enumerate(top5):
                    self.top5_widgets[i][0].config(text=f"{char.upper()}: {prob:.1f}%")
                    self.top5_widgets[i][1]['value'] = prob

            self.backend.img_initial = frame_gray_resized

            display_orig = cv.cvtColor(frame_gray_resized, cv.COLOR_GRAY2RGB)
            if self.backend.is_recording:
                cv.putText(display_orig, f"REC: {self.backend.current_target_class.upper()}", (5, 15), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

            img_orig, img_tc = ImageTk.PhotoImage(Image.fromarray(display_orig)), ImageTk.PhotoImage(Image.fromarray(tc_frame))
            self.lbl_orig.configure(image=img_orig)
            self.lbl_orig.image = img_orig
            self.lbl_tc.configure(image=img_tc)
            self.lbl_tc.image = img_tc

        self.root.after(30, self.process_video)