import skimage
from skimage import data 
import numpy as np
import matplotlib.pyplot as plt

print(skimage.__version__)

sample_image = np.random.random([300,300])

plt.imshow(sample_image)
plt.show()