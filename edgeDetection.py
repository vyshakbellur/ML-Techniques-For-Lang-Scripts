import skimage
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import convex_hull_image
from skimage.util import invert
from dsp.utils import get_corpus
from skimage import color, io
from skimage.filters import roberts, sobel

print(skimage.__version__)

# sample_image = np.random.random([300,300])

# plt.imshow(sample_image)

files = get_corpus("./DataSet/Devanagari/")

print("%d files"%(len(files)))

# al = skimage.io.imread("./DataSet/Croatian/A.png", as_gray=True)
# plt.imshow(al)
# print(al.shape)

for file in files[:4]:
    img = skimage.io.imread(file)
    imgJPG = img[:,:,:3]
    greyImg = color.rgb2gray(imgJPG)
    invertImg = invert(greyImg)
    roberts_img  = roberts(greyImg)
    plt.imshow(roberts_img, cmap='gray')
    sobel_img = sobel(greyImg)
    plt.imshow(roberts_img, cmap='gray')
    
    fig, axes = plt.subplots(1, 2, 
                         figsize=(12, 8), 
                         sharex=True, sharey=True)  
    ax = axes.ravel() 
    ax[0].set_title('roberts edge detection')
    ax[0].imshow(sobel_img, cmap='gray')
    ax[1].set_title('sobel edge detection')
    ax[1].imshow(roberts_img, cmap='gray')
    fig.tight_layout()
    plt.show()
    # convexhull_Img = convex_hull_image(invertImg) 

    # fig, axes = plt.subplots(1, 2, 
    #                      figsize=(12, 12), 
    #                      sharex=True, sharey=True) 
    # ax = axes.ravel()

    # ax[0].set_title('Inverted Image')
    # ax[0].imshow(invertImg,
    #             cmap='gray', 
    #             interpolation='nearest')  

    # ax[1].set_title('Convex Hull')
    # ax[1].imshow(convexhull_Img, 
    #             cmap='gray', 
    #             interpolation='nearest')  

    # plt.tight_layout()  
    plt.show()



plt.show()