import cv2 as cv
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from PIL import Image, ImageTk
from dvsl_backend import dvsl_backend

class dvsl_frontend:
    def __init__(self, root):
        self.root = root
        self.root.title("DVS ASL")
        self.dvsl_backend = dvsl_backend()
        
        self.camera = None
        self.stream_running = False
        self.is_testing = False
        self.raw_sequence = []
        
        self.setup_ui()

    def setup_ui(self):
        self.top_frame = ttk.Frame(self.root, padding=10)
        self.top_frame.pack(fill=tk.X)

        ttk.Label(self.top_frame, text="Camera:").pack(side=tk.LEFT)
        self.camera_var = tk.StringVar(value="0")
        self.camera_cb = ttk.Combobox(self.top_frame, textvariable=self.camera_var, values=["0", "1", "2", "3"], width=5)
        self.camera_cb.pack(side=tk.LEFT, padx=5)

        self.btn_stream = ttk.Button(self.top_frame, text="Start Stream", command=self.toggle_stream)
        self.btn_stream.pack(side=tk.LEFT, padx=5)

        self.btn_admin = ttk.Button(self.top_frame, text="Admin Access", command=self.request_admin)
        self.btn_admin.pack(side=tk.RIGHT)

        self.video_frame = ttk.Frame(self.root, padding=10)
        self.video_frame.pack()

        self.lbl_orig = ttk.Label(self.video_frame)
        self.lbl_orig.pack(side=tk.LEFT, padx=10)
        
        self.lbl_tc = ttk.Label(self.video_frame)
        self.lbl_tc.pack(side=tk.LEFT, padx=10)

        self.container_frame = ttk.Frame(self.root, padding=10)
        self.container_frame.pack(fill=tk.BOTH, expand=True)

        self.user_frame = ttk.Frame(self.container_frame)
        self.admin_frame = ttk.Frame(self.container_frame)
        
        self.setup_user_view()
        self.setup_admin_view()
        
        self.user_frame.pack(fill=tk.BOTH, expand=True)

    def setup_user_view(self):
        ttk.Label(self.user_frame, text="Realtime ASL Classifier", font=('', 14, 'bold')).pack(pady=5)
        
        self.btn_test = ttk.Button(self.user_frame, text="Start Predicting", command=self.toggle_testing)
        self.btn_test.pack(pady=5)

        ttk.Label(self.user_frame, text="Current Sign:").pack()
        
        self.lbl_current_pred = ttk.Label(self.user_frame, text="-", font=('', 64, 'bold'), foreground="blue")
        self.lbl_current_pred.pack(pady=20)

    def setup_admin_view(self):
        ttk.Label(self.admin_frame, text="Admin Control Panel", font=('', 14, 'bold')).pack(pady=5)

        ttk.Button(self.admin_frame, text="Exit Admin", command=self.show_user_view).pack(pady=5)

        data_frame = ttk.LabelFrame(self.admin_frame, text="Data Collection", padding=10)
        data_frame.pack(fill=tk.X, pady=5)

        ttk.Label(data_frame, text="Class:").pack(side=tk.LEFT)
        self.class_var = tk.StringVar(value="a")
        self.class_cb = ttk.Combobox(data_frame, textvariable=self.class_var, values=self.dvsl_backend.classes, width=5)
        self.class_cb.pack(side=tk.LEFT, padx=5)

        self.btn_record = ttk.Button(data_frame, text="Start Recording", command=self.toggle_recording)
        self.btn_record.pack(side=tk.LEFT, padx=5)

        train_frame = ttk.LabelFrame(self.admin_frame, text="Model Pipeline", padding=10)
        train_frame.pack(fill=tk.X, pady=5)

        ttk.Button(train_frame, text="Load Model", command=self.dvsl_backend.load_model).pack(side=tk.LEFT, padx=5)

    def request_admin(self):
        pwd = simpledialog.askstring("Admin Login", "Enter Password:", show='*')
        if pwd == "admin":
            self.user_frame.pack_forget()
            self.admin_frame.pack(fill=tk.BOTH, expand=True)
        elif pwd is not None:
            messagebox.showerror("Error", "Incorrect Password")

    def show_user_view(self):
        self.admin_frame.pack_forget()
        self.user_frame.pack(fill=tk.BOTH, expand=True)

    def toggle_stream(self):
        if self.stream_running:
            self.stream_running = False
            self.btn_stream.config(text="Start Stream")
            if self.camera:
                self.camera.release()
                self.camera = None
            self.dvsl_backend.img_initial_gray_resized = None
        else:
            cam_idx = int(self.camera_var.get())
            self.camera = cv.VideoCapture(cam_idx)
            if self.camera.isOpened():
                self.stream_running = True
                self.btn_stream.config(text="Stop Stream")
                self.process_video()
            else:
                messagebox.showerror("Error", "Could not open camera")

    def toggle_recording(self):
        self.dvsl_backend.is_recording = not self.dvsl_backend.is_recording
        if self.dvsl_backend.is_recording:
            self.dvsl_backend.current_target_class = self.class_var.get()
            self.btn_record.config(text="Stop Recording")
        else:
            self.btn_record.config(text="Start Recording")

    def toggle_testing(self):
        self.is_testing = not self.is_testing
        if self.is_testing:
            self.btn_test.config(text="Stop Predicting")
            self.dvsl_backend.clear_buffer() # Reset the sliding window
            self.lbl_current_pred.config(text="-")
        else:
            self.btn_test.config(text="Start Predicting")
            self.lbl_current_pred.config(text="-")

    def process_video(self):
        if not self.stream_running or self.camera is None:
            return

        ret, frame = self.camera.read()
        if ret:
            frame = cv.flip(frame, 1)
            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            frame_gray_resized = self.dvsl_backend.resize_frame(frame_gray)

            if self.dvsl_backend.img_initial_gray_resized is None:
                self.dvsl_backend.img_initial_gray_resized = frame_gray_resized

            tc_frame = self.dvsl_backend.temporal_contrast(self.dvsl_backend.img_initial_gray_resized, frame_gray_resized)

            if self.dvsl_backend.is_recording:
                self.dvsl_backend.save_frame(tc_frame)

            if self.is_testing:
                pred = self.dvsl_backend.predict_character(tc_frame)
                self.lbl_current_pred.config(text=pred.upper())

            self.dvsl_backend.img_initial_gray_resized = frame_gray_resized

            display_orig = cv.cvtColor(frame_gray_resized, cv.COLOR_GRAY2RGB)
            if self.dvsl_backend.is_recording:
                cv.putText(display_orig, f"REC: {self.dvsl_backend.current_target_class.upper()}", (5, 15), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

            img_orig = ImageTk.PhotoImage(image=Image.fromarray(display_orig))
            img_tc = ImageTk.PhotoImage(image=Image.fromarray(tc_frame))

            self.lbl_orig.imgtk = img_orig
            self.lbl_orig.configure(image=img_orig)
            
            self.lbl_tc.imgtk = img_tc
            self.lbl_tc.configure(image=img_tc)

        self.root.after(30, self.process_video)