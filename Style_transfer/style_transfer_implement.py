# %load_ext autoreload
# %autoreload 2
from Style_transfer_vgg19 import *  # include imread, imsave
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import scipy.misc
import matplotlib.gridspec as gridspec

style_scale = 1
width = 224

cont_img = imread('styles/city.jpg', width = width)
style_images = []
style_images.append(imread('styles/starry_night.jpg', width = style_scale*width))
style_blend_weights = [1.0/len(style_images) for _ in style_images]

# first condition
new_images1 = []
new_img = None

tf.reset_default_graph()
with tf.device('/gpu:0'):
    new_img =  stylize(initial = None, initial_noiseblend = 1.0,  # to use specific image  provide image to initial and set noiseblend
                content= cont_img, styles = style_images, style_blend_weights = style_blend_weights,
                preserve_colors = True, network = 'imagenet-vgg-verydeep-19.mat',
                content_weight=5e0, content_weight_blend=1, style_weight = 5e2, style_layer_weight_exp = 1,tv_weight = 1e2,
                learning_rate = 1e1, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, pooling = 'max',
                iterations = 500, print_iterations=500, checkpoint_iterations=None)
    for iteration, image in new_img :
        new_images1.append(image)

#second image
new_images2 = []
new_img = None

tf.reset_default_graph()
with tf.device('/gpu:0'):
    new_img =  stylize(initial = cont_img, initial_noiseblend = 0,  # to use specific image  provide image to initial and set noiseblend
                content= cont_img, styles = style_images, style_blend_weights = style_blend_weights,
                preserve_colors = False, network = 'imagenet-vgg-verydeep-19.mat',
                content_weight=5e0, content_weight_blend=0.5, style_weight = 5e2, style_layer_weight_exp = 1,tv_weight = 1e2,
                learning_rate = 1e1, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, pooling = 'max',
                iterations = 500, print_iterations=500, checkpoint_iterations=None)
    for iteration, image in new_img :
        new_images2.append(image)

#show image
plt.rcParams['figure.figsize'] = (12, 10)

plt.subplot(2, 2, 1)
plt.axis('off')
plt.title('content image', fontsize=20)
plt.imshow(np.clip(cont_img, 0, 255).astype(np.uint8))

plt.subplot(2, 2, 2)
plt.axis('off')
plt.title('style image', fontsize=20)
plt.imshow(np.clip(style_images[0], 0, 255).astype(np.uint8))

plt.subplot(2, 2, 3)
plt.imshow(np.clip(new_images1[0], 0, 255).astype(np.uint8))
plt.title("init : x, pooling : max", fontsize=20)
plt.axis('off')

plt.subplot(2, 2, 4)
plt.title("init : o, pooling : max", fontsize=20)
plt.imshow(np.clip(new_images2[0], 0, 255).astype(np.uint8))
plt.axis('off')

plt.show()

# save images
imsave('styles/saved_image/city_starrynight_preserve_color.jpg',new_images1[0])
imsave('styles/saved_image/city_starrynight.jpg',new_images2[0])
