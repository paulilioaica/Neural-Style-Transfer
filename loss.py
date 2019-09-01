import torch
import numpy as np


def gram(tensor):
    return torch.mm(tensor, tensor.t())


def gram_loss(g_matrix_style, g_matrix_content, N, M):


    return torch.sum(torch.pow(g_matrix_content- g_matrix_style, 2)) / (np.power(N * M * 2, 2, dtype=np.float64))


def total_variation_loss(image):
    loss = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])) + \
           torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]))
    return loss


def current_content_loss(noise, image):
    return 1 / 2. * torch.sum(torch.pow(noise - image, 2))
