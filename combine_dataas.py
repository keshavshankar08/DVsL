import shutil
from pathlib import Path

def merge_user_data():
    # Define the root directory (current folder)
    base_path = Path('.')
    
    # Define the target output directory
    output_base = base_path / 'data'
    
    # 1. Find all folders starting with 'data_' (excluding the output folder itself)
    user_folders = [f for f in base_path.iterdir() if f.is_dir() and f.name.startswith('data_')]
    
    if not user_folders:
        print("No folders starting with 'data_' were found.")
        return

    print(f"Found {len(user_folders)} source folders: {[f.name for f in user_folders]}")

    # 2. Iterate through each user folder (data_jay, data_keshav, etc.)
    for user_folder in user_folders:
        # Iterate through each sub-class folder (a, b, ..., z, none)
        for class_folder in user_folder.iterdir():
            if class_folder.is_dir():
                # Create the corresponding sub-folder in the new 'data/' directory
                target_dir = output_base / class_folder.name
                target_dir.mkdir(parents=True, exist_ok=True)
                
                # 3. Move/Copy all images from the source class to the target class
                for img in class_folder.glob('*.png'):
                    # Check if file exists to avoid overwriting identical timestamps
                    destination = target_dir / img.name
                    
                    if not destination.exists():
                        shutil.copy2(img, destination)
                    else:
                        # Handle collision: If two users have the exact same timestamp
                        new_name = f"{user_folder.name}_{img.name}"
                        shutil.copy2(img, target_dir / new_name)

    print(f"Successfully merged data into the '{output_base}' folder.")

if __name__ == "__main__":
    merge_user_data()