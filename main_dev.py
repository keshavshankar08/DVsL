from dvsl_frontend import dvsl_frontend
import tkinter as tk

if __name__ == "__main__":
    root = tk.Tk()
    app = dvsl_frontend(root)
    root.protocol("WM_DELETE_WINDOW", root.quit)
    root.mainloop()