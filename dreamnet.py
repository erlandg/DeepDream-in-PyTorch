import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision import models


class DD:
    def __init__(self, class_id = None):
        self.model = models.vgg16(pretrained = True)
        self.modules = list(self.model.features.modules())[1:]
        self.modules.extend(list(self.model.avgpool.modules()))
        self.modules.extend(list(self.model.classifier.modules())[1:])

        self.class_id = class_id
    
        self.imgSize = 1500

        self.Tmean = [0.485, 0.456, 0.406]
        self.Tsd = [0.229, 0.224, 0.225]
        self.Tnorm = T.Normalize(
            mean = self.Tmean,
            std = self.Tsd
        )

        self.Tresize = T.Resize((self.imgSize, self.imgSize))
        self.Tproc = T.Compose([
            T.ToTensor(),
            self.Tnorm
        ])

        self.Tmean = torch.Tensor(self.Tmean)
        self.Tsd = torch.Tensor(self.Tsd)
        
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.Tmean = self.Tmean.cuda()
            self.Tsd = self.Tsd.cuda()


    def toImg(self, inp):
        return inp*self.Tsd + self.Tmean


    def DeepDream(self, image, layer, iterations, lr):
        input = torch.autograd.Variable(image, requires_grad=True)

        self.model.zero_grad()
        for _ in range(iterations):
            out = input
            for Lid in range(layer + 1):
                if type(self.modules[Lid - 1]) == nn.modules.pooling.AdaptiveAvgPool2d:
                    out = torch.flatten(out, 1)
                out = self.modules[Lid](out)
                
            if (self.class_id != None) and (self.class_id != 'none'):
                loss = torch.zeros_like(out)
                loss[0, self.class_id] = 1
                out.backward(gradient=loss)
            else:
                loss = F.relu(out.norm())
                loss.backward()
                
            input.data = input.data + lr*torch.tanh(input.grad.data)

        return input


    def Recursive(self, image, layer, iterations, lr, num_downscales, dc_scale):
        image_ = self.Tresize(image) if sum([_ > self.imgSize for _ in image.size]) else image # if larger than cap
        _ = self.Tproc(image_).unsqueeze(0)
        if torch.cuda.is_available():
            _ = _.cuda()
        octaves = [_]

        for _ in range(num_downscales):
            octaves.append(nn.functional.interpolate(
                octaves[-1],
                scale_factor=(1.0/dc_scale, 1.0/dc_scale),
                recompute_scale_factor=False,
                mode='bilinear',
                align_corners=False
            ))

        detail = torch.zeros_like(octaves[-1])

        for octave, octave_base in enumerate(octaves[::-1]):
            h, w = octave_base.shape[-2:]
            if octave > 0:
                detail = nn.functional.interpolate(
                    detail,
                    size=(h, w),
                    mode='bilinear',
                    align_corners=False
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
        self.model.eval()
        if self.class_id == 'random':
            try:
                self.class_id = torch.randint(high=self.modules[LAYER_ID].out_features, size=(1,))
            except AttributeError:
                print('class_id = "random" can only be applied on linear layers, continuing with None')
                self.class_id = None
        return self.Recursive(image, LAYER_ID, NUM_ITERATIONS, LR, NUM_DOWNSCALES, SCALE).cpu()



if __name__ == '__main__':
    for i, module in enumerate(DD().modules):
        print(f'Layer_ID : {i} : {module}')
