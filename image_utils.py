from PIL import Image
IMAGE_SIZE = (768, 1000)

def load_image(path):
    img = Image.open(path)
    img = img.resize(IMAGE_SIZE)
    return img
