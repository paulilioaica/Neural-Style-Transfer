from PIL import Image
IMAGE_SIZE = (256, 256)

def load_image(path):
    img = Image.open(path)
    img = img.resize(IMAGE_SIZE)
    return img
