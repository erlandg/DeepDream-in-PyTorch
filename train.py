from dreamnet import *
from torchvision.utils import save_image
import os


def white_noise_image(w,h):
    bw_map = Image.fromarray(np.random.randint(126,127,(w,h,3),dtype=np.dtype('uint8')))
    return bw_map


LAYER_ID = 28 # Last conv layer
LAYER_ID = 14 # "Eye" layer

NUM_ITERATIONS = 5
LR = .5

NUM_DOWNSCALES = 4
SCALE = 1.4

REPEAT = False
NUM_REPEAT = 20


dir_pth = os.getcwd()
fname = 'hengekoye'
ftype = 'jpg'
IMAGE_PATH = dir_pth+'\\imgs\\%s.%s' % (fname, ftype)
SAVE_PATH = lambda idx: dir_pth+'\\outs\\%s_%s.%s' % (fname, str(idx), ftype)


img = Image.open(IMAGE_PATH)
# img = img.rotate(270, Image.NEAREST, expand = 1) # If vertical

# img = white_noise_image(1500, 1500)

DeepImg = DD().Run(img, LAYER_ID, NUM_ITERATIONS, LR, NUM_DOWNSCALES, SCALE)

if REPEAT:
    for i in range(NUM_REPEAT):
        DeepRep = T.functional.to_pil_image(DeepImg.permute(2,0,1))
        DeepImg = DD().Run(DeepRep, LAYER_ID, NUM_ITERATIONS, LR, NUM_DOWNSCALES, SCALE)

i = 1
while i:
    if not os.path.isfile(SAVE_PATH(i)):
        save_image(DeepImg.permute(2,0,1), SAVE_PATH(i))
        i = 0
    else: i += 1

plt.imshow(DeepImg)
plt.show()
