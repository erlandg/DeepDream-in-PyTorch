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


def output_fname(fname, index, ftype):
    return 'out/{}_{}.{}'.format(fname, index, ftype)


def read_dict_from_txt(filepath):
    f = open(filepath, 'r').read()
    return eval(f)


@click.command()
@click.argument('filepath', type=click.Path(exists=True))
@click.option(
    '-r', '--repeat', default=0,
    help = 'The number of times to repeat the process (default = 0)'
)
@click.option(
    '-s', '--single-classes', is_flag=True, flag_value='random', default=False,
    help = 'Whether to set gradients as loss, or maximise one-and-one neuron at a time'
)
@click.option(
    '-d', '--display-image', is_flag=True, default=False,
    help = 'Boolean, whether to display output image or not (default = False)'
)
def main(filepath, repeat, single_classes, display_image):

    labels = read_dict_from_txt('labels.txt')

    *fname, ftype = filepath.split('/')[-1].split('.')
    fname = '.'.join(fname)

    img = Image.open(filepath)

    print('Dreaming ...')
    DD_ = DD(single_classes)
    DeepImg = DD_.Run(
        img,
        config.LAYER_ID,
        config.NUM_ITERATIONS,
        config.LR,
        config.NUM_DOWNSCALES,
        config.SCALE
    )
    if (config.LAYER_ID == len(DD_.modules)-1) and (single_classes == 'random'):
        print(f'Dreamt of {labels[int(DD_.class_id)]} ...')

    for i in range(repeat):
        DeepRep = T.functional.to_pil_image(DeepImg.permute(2,0,1))
        DD_ = DD(single_classes)
        DeepImg = DD_.Run(
            DeepRep,
            config.LAYER_ID,
            config.NUM_ITERATIONS,
            config.LR,
            config.NUM_DOWNSCALES,
            config.SCALE
        )
        if (config.LAYER_ID == len(DD_.modules)-1) and (single_classes == 'random'):
            print(f'Dreamt of {labels[int(DD_.class_id)]} ...')


    print('Saving ...')
    if not os.path.isdir('out'):
        os.mkdir('out')

    i = 1
    while os.path.isfile(output_fname(fname, i, ftype)):
        i += 1

    save_image(DeepImg.permute(2,0,1), output_fname(fname, i, ftype))
    print('Image saved.')


    if display_image:
        plt.imshow(DeepImg)
        plt.show()



if __name__ == "__main__":
    main()