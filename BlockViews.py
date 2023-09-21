import skimage.io
from skimage import data 
from skimage import color
from skimage.util import view_as_blocks

import numpy as np
import matplotlib.pyplot as plt

devaNag = skimage.io.imread("./DataSet/Devanagari/a.png")[:,:96,:3]
devaNagGray = color.rgb2gray(devaNag)
# plt.imshow(devaNag)

print(" devNag Shape ", devaNag.shape)
print(" devNag  gray Shape ", devaNagGray.shape)
block_shape = (2,2)
devaNag_blocks = view_as_blocks(devaNagGray, block_shape)

print(" devNag block Shape ", devaNag_blocks.shape)


plt.show()
