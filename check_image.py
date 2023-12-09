from PIL import Image

def check_image_size(image_path, target_width, target_height):
    # Open the image
    img = Image.open(image_path)

    # Get the size of the image
    width, height = img.size

    # Check if the image has the target size
    return (width == target_width and height == target_height)