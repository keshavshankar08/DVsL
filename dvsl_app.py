import customtkinter as ctk
import cv2 as cv
from PIL import Image
import threading
from dvsl_backend import dvsl_backend
from profile_manager import ProfileManager

# Set sleek modern theme
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class DVSLApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("DVS ASL Learning")
        self.geometry("900x700")
        
        # Backend & Managers
        self.profile_mgr = ProfileManager()
        self.backend = dvsl_backend()
        
        # State Variables
        self.current_user = None
        self.words_spelled = 0
        self.camera = None
        self.stream_running = False
        self.current_word = ""
        self.current_letter_idx = 0
        self.letter_labels = []
        
        # Cleanup on exit
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.show_login_screen()

    # --- SCREEN 1: LOGIN ---
    def show_login_screen(self):
        self._clear_window()
        
        frame = ctk.CTkFrame(self, fg_color="transparent")
        frame.place(relx=0.5, rely=0.5, anchor="center")
        
        ctk.CTkLabel(frame, text="Welcome to ASL Learner", font=("Helvetica", 28, "bold")).pack(pady=20)
        
        users = self.profile_mgr.get_users()
        user_list = list(users.keys()) if users else ["No users found"]
        
        self.user_var = ctk.StringVar(value=user_list[0] if user_list else "")
        self.user_dropdown = ctk.CTkOptionMenu(frame, variable=self.user_var, values=user_list, width=200)
        self.user_dropdown.pack(pady=10)
        
        ctk.CTkButton(frame, text="Load Profile", command=self.load_profile, width=200).pack(pady=10)
        
        ctk.CTkLabel(frame, text="Or create a new profile:", font=("Helvetica", 14)).pack(pady=(20, 5))
        self.new_user_entry = ctk.CTkEntry(frame, placeholder_text="Enter name...", width=200)
        self.new_user_entry.pack(pady=5)
        
        ctk.CTkButton(frame, text="Create Profile", command=self.create_profile, width=200, fg_color="green", hover_color="darkgreen").pack(pady=10)

    def load_profile(self):
        user = self.user_var.get()
        if user and user != "No users found":
            self.current_user = user
            self.words_spelled = self.profile_mgr.get_users().get(user, 0)
            self.show_setup_screen()

    def create_profile(self):
        new_name = self.new_user_entry.get().strip()
        if new_name:
            self.current_user = new_name
            self.words_spelled = 0
            self.profile_mgr.save_user_progress(new_name, 0)
            self.show_setup_screen()

    # --- SCREEN 2: SETUP ---
    def show_setup_screen(self):
        self._clear_window()
        
        frame = ctk.CTkFrame(self, fg_color="transparent")
        frame.place(relx=0.5, rely=0.5, anchor="center")
        
        ctk.CTkLabel(frame, text=f"Profile: {self.current_user}", font=("Helvetica", 24, "bold")).pack(pady=5)
        ctk.CTkLabel(frame, text=f"Total Words Spelled: {self.words_spelled}", font=("Helvetica", 16), text_color="gray").pack(pady=(0, 30))
        
        ctk.CTkLabel(frame, text="Select Camera:", font=("Helvetica", 14)).pack()
        self.cam_var = ctk.StringVar(value="0")
        ctk.CTkOptionMenu(frame, variable=self.cam_var, values=["0", "1", "2", "3"], width=200).pack(pady=10)
        
        ctk.CTkLabel(frame, text="Select Difficulty:", font=("Helvetica", 14)).pack(pady=(20,0))
        self.diff_var = ctk.StringVar(value="Easy")
        ctk.CTkOptionMenu(frame, variable=self.diff_var, values=["Easy", "Medium", "Hard"], width=200).pack(pady=10)
        
        ctk.CTkButton(frame, text="Start Game", command=self.start_game, width=200, height=40, font=("Helvetica", 16, "bold")).pack(pady=30)
        
        ctk.CTkButton(frame, text="Switch User", command=self.show_login_screen, fg_color="transparent", border_width=1, text_color="gray").pack()

    # --- SCREEN 3: GAMEPLAY ---
    def start_game(self):
        # Init Camera
        cam_idx = int(self.cam_var.get())
        self.camera = cv.VideoCapture(cam_idx)
        if not self.camera.isOpened():
            print("Failed to open camera!")
            return
            
        self.stream_running = True
        self._clear_window()
        
        # --- UI Layout ---
        # Top: Stats & Wordle Frame
        top_frame = ctk.CTkFrame(self, fg_color="transparent")
        top_frame.pack(fill="x", pady=20, padx=20)
        
        self.lbl_stats = ctk.CTkLabel(top_frame, text=f"Words Spelled: {self.words_spelled}", font=("Helvetica", 16, "bold"), text_color="green")
        self.lbl_stats.pack(side="left")
        
        self.wordle_frame = ctk.CTkFrame(top_frame, fg_color="transparent")
        self.wordle_frame.pack(side="top", expand=True)
        
        # Middle: Video Streams
        vid_frame = ctk.CTkFrame(self, fg_color="transparent")
        vid_frame.pack(expand=True)
        
        self.lbl_orig = ctk.CTkLabel(vid_frame, text="")
        self.lbl_orig.pack(side="left", padx=10)
        self.lbl_tc = ctk.CTkLabel(vid_frame, text="")
        self.lbl_tc.pack(side="left", padx=10)
        
        # Bottom: Controls
        bot_frame = ctk.CTkFrame(self, fg_color="transparent")
        bot_frame.pack(side="bottom", pady=20)
        
        ctk.CTkButton(bot_frame, text="Skip Word", command=self.load_new_word, fg_color="orange", hover_color="darkorange").pack(side="left", padx=10)
        ctk.CTkButton(bot_frame, text="End Game", command=self.end_game, fg_color="red", hover_color="darkred").pack(side="left", padx=10)
        
        # Start Logic
        self.load_new_word()
        self.process_video_loop()

    def load_new_word(self):
        diff = self.diff_var.get()
        self.current_word = self.profile_mgr.get_random_word(diff)
        self.current_letter_idx = 0
        self.backend.clear_buffer()
        self.render_wordle_boxes()

    def render_wordle_boxes(self):
        # Clear old boxes
        for widget in self.wordle_frame.winfo_children():
            widget.destroy()
        self.letter_labels.clear()
        
        for i, char in enumerate(self.current_word):
            # Styling states
            if i < self.current_letter_idx:
                bg = "green"
                border = 0
            elif i == self.current_letter_idx:
                bg = "transparent"
                border = 2
            else:
                bg = "gray20"
                border = 0
                
            lbl = ctk.CTkLabel(self.wordle_frame, text=char.upper(), font=("Helvetica", 32, "bold"), 
                               width=60, height=60, corner_radius=10, 
                               fg_color=bg, text_color="white")
            
            # CustomTkinter doesn't have a direct 'border_color' for Labels easily, 
            # so we use a sub-frame trick if it's the active letter to give it a border
            if i == self.current_letter_idx:
                border_frame = ctk.CTkFrame(self.wordle_frame, border_width=3, border_color="#1f538d", corner_radius=10, fg_color="transparent")
                border_frame.pack(side="left", padx=5)
                lbl = ctk.CTkLabel(border_frame, text=char.upper(), font=("Helvetica", 32, "bold"), width=54, height=54)
                lbl.pack(padx=3, pady=3)
                self.letter_labels.append(lbl) # Reference to inner label
            else:
                lbl.pack(side="left", padx=5)
                self.letter_labels.append(lbl)

    def handle_correct_letter(self):
        self.current_letter_idx += 1
        self.backend.clear_buffer() # Reset prediction wiggles
        
        if self.current_letter_idx >= len(self.current_word):
            # Word Complete!
            self.words_spelled += 1
            self.lbl_stats.configure(text=f"Words Spelled: {self.words_spelled}")
            self.profile_mgr.save_user_progress(self.current_user, self.words_spelled)
            # Add slight delay so user sees full green word before swapping
            self.after(1000, self.load_new_word)
        
        self.render_wordle_boxes()

    def process_video_loop(self):
        if not self.stream_running or self.camera is None:
            return

        ret, frame = self.camera.read()
        if ret:
            frame = cv.flip(frame, 1)
            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            frame_gray_resized = self.backend.resize_frame(frame_gray)

            if self.backend.img_initial_gray_resized is None:
                self.backend.img_initial_gray_resized = frame_gray_resized

            tc_frame = self.backend.temporal_contrast(self.backend.img_initial_gray_resized, frame_gray_resized)

            # GAME LOGIC: Check prediction
            pred = self.backend.predict_character(tc_frame)
            if self.current_letter_idx < len(self.current_word):
                target_char = self.current_word[self.current_letter_idx].lower()
                if pred == target_char:
                    self.handle_correct_letter()

            self.backend.img_initial_gray_resized = frame_gray_resized

            # Update UI Video Feeds
            display_orig = cv.cvtColor(frame_gray_resized, cv.COLOR_GRAY2RGB)
            img_orig = ctk.CTkImage(light_image=Image.fromarray(display_orig), size=(256, 256))
            img_tc = ctk.CTkImage(light_image=Image.fromarray(tc_frame), size=(256, 256))

            self.lbl_orig.configure(image=img_orig)
            self.lbl_tc.configure(image=img_tc)

        self.after(30, self.process_video_loop)

    def end_game(self):
        self.stream_running = False
        if self.camera:
            self.camera.release()
            self.camera = None
        self.backend.img_initial_gray_resized = None
        self.show_setup_screen()

    def _clear_window(self):
        for widget in self.winfo_children():
            widget.destroy()

    def on_closing(self):
        self.stream_running = False
        if self.camera:
            self.camera.release()
        import torch
        torch.cuda.empty_cache() # Clean up VRAM
        self.destroy()