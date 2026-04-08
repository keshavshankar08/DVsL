import sys
import os

def resource_path(relative_path):
    """Get the absolute path to a resource in a bundled app."""
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)