import os
import platform
import shutil

def build():
    system = platform.system()

    # Clean old build/dist folders first
    if os.path.exists('build'):
        shutil.rmtree('build')
    if os.path.exists('dist'):
        shutil.rmtree('dist')

    # Common base command
    cmd = 'pyinstaller --noconfirm --onefile --windowed '

    # Add Host.ui resource
    if system == "Darwin":
        cmd += '--name "TCASL" '
        cmd += '--add-data "tcasl.ui:." '
        cmd += '--icon tcasl.icns '
    elif system == "Windows":
        cmd += '--name "TCASL" '
        cmd += '--add-data "tcasl.ui;." '
        cmd += '--icon tcasl.ico '
    else:
        print(f"Unsupported OS: {system}")
        return

    # Final part of the command (the script name)
    cmd += 'app.py'

    # Run it
    os.system(cmd)

if __name__ == "__main__":
    build()