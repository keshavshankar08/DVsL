import tkinter as tk
from frontend import TCASLFrontend

if __name__ == "__main__":
    root = tk.Tk()
    app = TCASLFrontend(root)
    root.protocol("WM_DELETE_WINDOW", root.quit)
    root.mainloop()