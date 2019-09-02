from PIL import Image
IMAGE_SIZE = (1024, 1536)

def load_image(path):
    img = Image.open(path)
    img = img.resize(IMAGE_SIZE)
    return img
