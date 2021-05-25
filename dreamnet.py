import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as T
import numpy as np
import scipy.ndimage as nd
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter, ImageChops


class DD:
    def __init__(self):
        self.model = models.vgg16(pretrained = True)
        self.model = self.model.cuda()
        self.modules = list(self.model.features.modules())

        imgSize = 1500 # 224

        self.Tmean = [0.485, 0.456, 0.406]
        self.Tsd = [0.229, 0.224, 0.225]
        self.Tnorm = T.Normalize(
            mean = self.Tmean,
            std = self.Tsd
        )

        self.Tproc = T.Compose([
            T.Resize((imgSize, imgSize)),
            T.ToTensor(),
            self.Tnorm
        ])

        self.Tmean = torch.Tensor(self.Tmean).cuda()
        self.Tsd = torch.Tensor(self.Tsd).cuda()


    def toImg(self, inp):
        return inp*self.Tsd + self.Tmean



    def DeepDream(self, image, layer, iterations, lr):
        input = torch.autograd.Variable(image, requires_grad=True)

        self.model.zero_grad()
        for _ in range(iterations):
            out = input
            for Lid in range(layer):
                out = self.modules[Lid + 1](out)
            loss = out.norm()
            loss.backward()
            input.data = input.data + lr*input.grad.data

        return input


    def Recursive(self, image, layer, iterations, lr, num_downscales, dc_scale):
        octaves = [self.Tproc(image).unsqueeze(0).cuda()]

        for i in range(num_downscales - 1):
            octaves.append(nn.functional.interpolate(
                octaves[-1],
                scale_factor=(1.0/dc_scale, 1.0/dc_scale),
                mode='bilinear'
            ))

        detail = torch.zeros_like(octaves[-1])
        for octave, octave_base in enumerate(octaves[::-1]):
            h, w = octave_base.shape[-2:]
            if octave > 0:
                h1, w1 = detail.shape[-2:]
                detail = nn.functional.interpolate(
                    detail,
                    size=(h, w),
                    mode='bilinear'
                )

            out = octave_base + detail

            out = self.DeepDream(out, layer, iterations, lr)

            detail = out - octave_base


        out = nn.functional.interpolate(out, size=image.size[::-1]).squeeze()
        out.transpose_(0,1)
        out.transpose_(1,2)
        out = torch.clip(self.toImg(out), 0, 1)
        return out.detach()


    def Run(self, image, LAYER_ID, NUM_ITERATIONS, LR, NUM_DOWNSCALES, SCALE):
        return self.Recursive(image, LAYER_ID, NUM_ITERATIONS, LR, NUM_DOWNSCALES, SCALE).cpu()


if __name__ == '__main__':
    print(DD().model)
