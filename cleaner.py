import os
import shutil

def delete_folders_with_few_files(root_folder, max_files):
    folders = [f for f in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, f))]

    for folder in folders:
        folder_path = os.path.join(root_folder, folder)
        
        files_count = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])

        if files_count <= max_files:
            shutil.rmtree(folder_path)
            print(f"Deleted folder: {folder_path}")

delete_folders_with_few_files("data", 500)
