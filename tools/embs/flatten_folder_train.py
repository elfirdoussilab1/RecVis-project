import os
import shutil

# Ay: With this file, I want to flatten the CIRR train folder so that we can run save_blip_embds_imgs.
def flatten_folder(parent_folder):
    # List all subfolders in the parent folder
    subfolders = [os.path.join(parent_folder, d) for d in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, d))]

    for subfolder in subfolders:
        # Iterate through each item in the subfolder
        for item in os.listdir(subfolder):
            item_path = os.path.join(subfolder, item)
            # Move the file or folder to the parent folder
            shutil.move(item_path, parent_folder)

        # Remove the now-empty subfolder
        os.rmdir(subfolder)

# Example usage
if __name__ == "__main__":
    flatten_folder('datasets/CIRR/images/train')