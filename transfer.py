import os

from torch import nn, optim
from torchvision import transforms
from torchvision.utils import save_image

from loss import *
from image_utils import load_image, IMAGE_SIZE
from vgg import VGG

CHANNEL_NUM = 3
EPOCHS = 10000
content_weight = 10
style_weight = 10000
variation_weight = 1000
event = 10

device = torch.device('cuda')
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

if __name__ == "__main__":
    print(os.getcwd())
    content_image = load_image("Neural-Style-Transfer/input/content.jpg")
    style_image = load_image("Neural-Style-Transfer/input/style.jpg")

    transform = transforms.Compose([transforms.Resize(IMAGE_SIZE[0], IMAGE_SIZE[1]), transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    content_image = transform(content_image).unsqueeze(0).to(device)
    style_image = transform(style_image).unsqueeze(0).to(device)
    cnn = VGG().to(device).eval()
    for name, child in cnn.vgg.named_children():
        if isinstance(child, nn.MaxPool2d):
            cnn.vgg[int(name)] = nn.AvgPool2d(kernel_size=2, stride=2)
        elif isinstance(child, nn.ReLU):
            cnn.vgg[int(name)] = nn.ReLU(inplace=False)

    for param in cnn.vgg.parameters():
        param.requires_grad = False

    content_features = cnn.get_content_features(content_image).detach().view(IMAGE_SIZE[0], -1)
    style_features = [feature.squeeze().view(feature.shape[1], -1).detach() for feature in
                      cnn.get_style_features(style_image)]
    style_grams = [gram(feature) for feature in style_features]

    noise = torch.randn(1, 3, IMAGE_SIZE[0], IMAGE_SIZE[1], requires_grad=True, device='cuda')
    adam = optim.Adam(params=[noise], lr=0.01, betas=(0.9, 0.999))

    ############## TRAINING LOOP ###############################

    for iteration in range(EPOCHS):

        adam.zero_grad()

        noise_content_features = cnn.get_content_features(noise).view(IMAGE_SIZE[0], -1)

        content_loss = current_content_loss(noise_content_features, content_features)

        noise_style_features = [feature.squeeze().view(feature.shape[1], -1) for feature in
                                cnn.get_style_features(noise)]

        noise_grams = [gram(feature) for feature in noise_style_features]

        style_loss = 0
        for i in range(len(style_features)):
            N, M = noise_style_features[i].shape[0], noise_style_features[i].shape[1]
            style_loss = style_loss + (gram_loss(noise_grams[i], style_grams[i], N, M) / 5.)
        style_loss = style_loss.to(device)

        variation_loss = total_variation_loss(noise)

        total_loss = content_weight * content_loss + style_weight * style_loss + variation_weight * variation_loss

        total_loss.backward()

        adam.step()

        if iteration % event == 0:
            print("Iteration: {}, Content Loss: {:.3f}, Style Loss: {:.3f}, Var Loss: {:.3f}".format(iteration,
                                                                                                     content_weight * content_loss.item(),
                                                                                                     style_weight * style_loss.item(),
                                                                                                     variation_weight * variation_loss.item()))

        if not os.path.exists('Neural-Style-Transfer/styletransfer/'):
            os.mkdir('Neural-Style-Transfer/styletransfer/')

        if iteration % event == 0:
            save_image(noise.detach(), filename='./generated/iter_{}.png'.format(iteration))
