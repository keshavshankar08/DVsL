import json
import os
import random

class ProfileManager:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.users_file = "users.json"
        self.dicts_dir = "dictionaries"
        self._ensure_setup()

    def _ensure_setup(self):        
        # Create empty users file if not exists
        if not os.path.exists(self.users_file):
            with open(self.users_file, 'w') as f:
                json.dump({}, f)

        # Generate default dictionaries if missing
        defaults = {
            "easy.txt": ["cat", "dog", "sun", "hat", "bat", "car", "cup", "pen"],
            "medium.txt": ["apple", "house", "train", "plane", "mouse", "water", "chair"],
            "hard.txt": ["elephant", "computer", "keyboard", "mountain", "building", "software"]
        }
        for filename, words in defaults.items():
            filepath = os.path.join(self.dicts_dir, filename)
            if not os.path.exists(filepath):
                with open(filepath, 'w') as f:
                    f.write("\n".join(words))

    def get_users(self):
        with open(self.users_file, 'r') as f:
            return json.load(f)

    def save_user_progress(self, username, words_spelled):
        users = self.get_users()
        users[username] = words_spelled
        with open(self.users_file, 'w') as f:
            json.dump(users, f, indent=4)

    def get_random_word(self, difficulty):
        filepath = os.path.join(self.dicts_dir, f"{difficulty.lower()}.txt")
        if not os.path.exists(filepath):
            return "error"
        with open(filepath, 'r') as f:
            words = [line.strip().lower() for line in f if line.strip()]
        return random.choice(words) if words else "empty"