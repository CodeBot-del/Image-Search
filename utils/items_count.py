import os
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Count Files in a Folder')
parser.add_argument('--folder', type=str, help='Path to the folder to count files in')
args = parser.parse_args()

# Function to count files in a folder
def count_files_in_folder(folder_path):
    if not os.path.exists(folder_path):
        print(f"The folder '{folder_path}' does not exist.")
        return
    
    file_count = 0
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_count += 1
    
    return file_count

# Count files in the specified folder
folder_path = args.folder
num_files = count_files_in_folder(folder_path)

if num_files is not None:
    print(f"There are {num_files} files in the folder '{folder_path}'.")
