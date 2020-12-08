import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
import os
from keras import backend as K
from keras.preprocessing.image import load_img, save_img, img_to_array
import matplotlib.pyplot as plt
from keras.applications import vgg19
from keras.models import Model

vgg19

#vgg19_weights = '../input/vgg19/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'
#vgg19 = VGG19(include_top = False, weights=vgg19_weights)


base_image_path = 'start.jpg'
style_image_path = 'model.jpg'
# Any results you write to the current directory are saved as output.
#fetch merge push


width, height = load_img(base_image_path).size
img_nrows = 400
img_ncols = int(width * img_nrows / height)
