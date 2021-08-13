from dreamnet import DD
import config

import torchvision.transforms as T
from torchvision.utils import save_image

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import os, click


def white_noise_image(w,h):
    bw_map = Image.fromarray(np.random.randint(126,127,(w,h,3),dtype=np.dtype('uint8')))
    return bw_map


@click.command()
@click.argument('filepath', type=click.Path(exists=True))
@click.option(
    '-r', '--repeat', default=0,
    help = 'The number of times to repeat the process (default = 0)'
)
@click.option(
    '-d', '--display-image', is_flag=True, default=False,
    help = 'Boolean, whether to display output image or not (default = False)'
)
def main(filepath, repeat, display_image):

    *fname, ftype = filepath.split('/')[-1].split('.')
    fname = '.'.join(fname)

    img = Image.open(filepath)

    print('Dreaming ...')
    DeepImg = DD().Run(
        img,
        config.LAYER_ID,
        config.NUM_ITERATIONS,
        config.LR,
        config.NUM_DOWNSCALES,
        config.SCALE
    )

    for i in range(repeat):
        DeepRep = T.functional.to_pil_image(DeepImg.permute(2,0,1))
        DeepImg = DD().Run(
            DeepRep,
            config.LAYER_ID,
            config.NUM_ITERATIONS,
            config.LR,
            config.NUM_DOWNSCALES,
            config.SCALE
        )

    print('Saving ...')
    i = 1
    while i:
        fpath = 'outs/{}_{}.{}'.format(fname, i, ftype)
        if not os.path.isfile(fpath):
            save_image(DeepImg.permute(2,0,1), fpath)
            print('Image saved.')
            i = 0
        else: i += 1


    if display_image:
        plt.imshow(DeepImg)
        plt.show()



if __name__ == "__main__":
    main()