#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Project      : ScreenShootResilient
# @File         : gan_tools.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2023/3/18 16:17

# Import lib here
import random
import torch


class ImagePool:
    """
    This class implements an image buffer that stores previously generated images! This buffer enables to update
    discriminators using a history of generated image rather than the latest ones produced by generator.
    """

    def __init__(self, pool_sz: int = 50):

        """
        Parameters:
            pool_sz: Size of the image buffer
        """

        self.nb_images = 0
        self.image_pool = []
        self.pool_sz = pool_sz

    def push_and_pop(self, images):

        """
        Parameters:
            images: latest images generated by the generator
        Returns a batch of images from pool!
        """

        if self.pool_sz == 0:
            return images

        images_to_return = []
        for image in images:
            image = torch.unsqueeze(image, 0)

            if self.nb_images < self.pool_sz:
                self.image_pool.append(image)
                images_to_return.append(image)
                self.nb_images += 1
            else:
                if random.uniform(0, 1) > 0.5:
                    rand_int = random.randint(0, self.pool_sz-1)
                    temp_img = self.image_pool[rand_int].clone()
                    self.image_pool[rand_int] = image
                    images_to_return.append(temp_img)
                else:
                    images_to_return.append(image)

        return torch.cat(images_to_return, 0)


def cal_gradient_penalty(discriminator, real_data, fake_data, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        discriminator (network)     -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( ||gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':  # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand_like(real_data)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = discriminator(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones_like(disc_interpolates),
                                        create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.reshape(real_data.shape[0], -1)
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp  # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


def run():
    pass


if __name__ == '__main__':
    run()
