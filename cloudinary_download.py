import os
import requests
from cloudinary import api, config

# Step 1: Configure Cloudinary API
config(
    cloud_name="dn9q44mjr",
    api_key="686952679495779",
    api_secret="cxROqIgoPfu785Qun5LceMDE-bE"
)

# Step 2: Function to List and Download All Images
def download_cloudinary_images():
    next_cursor = None
    download_folder = "cloudinary_images"
    os.makedirs(download_folder, exist_ok=True)

    while True:
        # Fetch image resources from Cloudinary
        # max_results=100 specifies the maximum number of resources to fetch in a single API call
        resources = api.resources(type="upload", resource_type="image", max_results=100, next_cursor=next_cursor)

        for resource in resources.get("resources", []):
            image_url = resource["url"]
            public_id = resource["public_id"]
            file_extension = image_url.split(".")[-1]

            # File path for download
            file_name = f"{public_id}.{file_extension}"
            file_path = os.path.join(download_folder, file_name)

            # Download the image
            print(f"Downloading {image_url}...")
            response = requests.get(image_url)
            with open(file_path, "wb") as file:
                file.write(response.content)

        # Check if there are more resources to fetch
        next_cursor = resources.get("next_cursor")
        if not next_cursor:
            break

    print(f"All images have been downloaded to the '{download_folder}' folder.")

# Step 3: Execute the Script
if __name__ == "__main__":
    download_cloudinary_images()
