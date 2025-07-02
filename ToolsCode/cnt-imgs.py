import os

# Define the path to your folder
folder_path = 'C://Users//HP//Documents//Projects//CAPTCHA_SOLVER//Scrambled copy'

# Define valid image extensions (you can add more if needed)
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp']

# Count image files
image_count = sum(
    1 for filename in os.listdir(folder_path)
    if os.path.splitext(filename)[1].lower() in image_extensions
)

print(f"Number of image files in the folder: {image_count}")
