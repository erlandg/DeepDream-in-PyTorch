from dreamnet import DD
import config

import torch
import torchvision.transforms as T
from torchvision.utils import save_image

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
from scipy import ndimage as nd
import imageio

import os, click, glob
from tqdm import tqdm


def true_if_dreaming_of_things(module_length, class_index):
    a = config.LAYER_ID == module_length-1
    b1 = type(class_index) == int
    b2 = class_index == 'random'
    return True if a and (b1 or b2) else False


def white_noise_image(w,h):
    # bw_map = Image.fromarray(np.random.randint(0,255,(w,h,3),dtype=np.dtype('uint8')))
    bw_map = Image.fromarray(np.random.normal(loc=128,scale=35,size=(h,w,3)).astype('uint8'))
    return bw_map


def output_fname(folder, fname, index, ftype):
    return '{}/{}_{:04d}.{}'.format(folder, fname, index, ftype)


def read_dict_from_txt(filepath):
    f = open(filepath, 'r').read()
    return eval(f)


def save_gif(path, save_path, fname='', duration=.02):
    images = []
    filenames = sorted(glob.glob(f'{path}/{fname}*'))
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave(save_path, images, duration=duration)



@click.command()
@click.argument(
    'filepath', type=click.Path()
)
@click.option(
    '-r', '--repeat', default=0,
    help = 'The number of times to repeat the process (default = 0)'
)
@click.option(
    '-s', '--single-classes', is_flag=False, flag_value='random', default=None,
    help = 'Whether to set gradients as loss, or maximise one-and-one neuron at a time'
)
@click.option(
    '-g', '--make-gif', is_flag=True, default=False,
    help = 'Whether to make a gif of the progression'
)
def main(filepath, repeat, single_classes, make_gif):

    try:
        single_classes = eval(single_classes)
    except (NameError, TypeError):
        pass

    labels = read_dict_from_txt('labels.txt')

    if filepath != 'none':
        *fname, ftype = filepath.split('/')[-1].split('.')
        fname = '.'.join(fname)
        img = Image.open(filepath)
    else:
        fname = 'white_noise'
        ftype = 'jpg'
        img = white_noise_image(
            config.WHITE_NOISE_WIDTH,
            config.WHITE_NOISE_HEIGHT
        )


    _ = 1
    while os.path.isdir(f'out_{_:02d}'):
        _ += 1
    folder_path = "out_{:02d}".format(_)
    os.mkdir(folder_path)


    print('Dreaming ...')
    DD_ = DD(single_classes)
    true_if_things = true_if_dreaming_of_things(len(DD_.modules), single_classes)

    DeepRep = img
    s = .01
    h, w = DeepRep.size
    for i in tqdm(range(repeat+1)):
        DeepImg = DD_.Run(
            DeepRep,
            config.LAYER_ID,
            config.NUM_ITERATIONS,
            config.LR,            config.NUM_DOWNSCALES,
            config.SCALE,
            reevaluate_class_id = single_classes == 'random'
        )
        DeepRep = Image.fromarray(
            (255*nd.affine_transform(
                DeepImg, [1-s,1-s,1], [h*s/2,w*s/2,0], order=1
            )).astype('uint8'), 'RGB'
        )

        if true_if_things:
            tqdm.write(f'Dreamt of {labels[int(DD_.class_id)]} ...')


        # print('Saving ...')
        save_image(DeepImg.permute(2,0,1), output_fname(folder_path, fname, i, ftype))
        # print('Image saved.')

    if make_gif:
        print('Making GIF ...')
        save_gif(folder_path,f'{folder_path}/DeepDream.gif',fname=fname)
        print('GIF saved.')



if __name__ == "__main__":
    main()
