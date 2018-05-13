import vgg

import tensorflow as tf
import numpy as np
from sys import stderr
from PIL import Image
import scipy.misc
import math
import matplotlib.pyplot as plt
from functools import reduce

CONTENT_LAYERS = ('relu1_1', 'relu1_2', 'relu2_1', 'relu2_2',
                  'relu3_1', 'relu3_2', 'relu3_3', 'relu3_4',
                  'relu4_1', 'relu4_2', 'relu4_3', 'relu4_4',
                  'relu5_1', 'relu5_2', 'relu5_3', 'relu5_4'
                  )

STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')

network = 'imagenet-vgg-verydeep-19.mat'

STYLE_SCALE = 1.0


def stylize(content, style,
            initial, initial_noiseblend,
            content_weight=5e0, content_layer_num=9,
            style_weight = 5e2, style_layer_weight = (0.2, 0.2, 0.2, 0.2, 0.2),
            tv_weight = 1e2, learning_rate = 1e1, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8,
            preserve_colors=False, pooling = 'max',
            iterations=1000, print_iterations=None, checkpoint_iterations=None):
    """
    Stylize images.
    This function yields tuples (iteration, image); `iteration` is None
    if this is the final image (the last iteration).  Other tuples are yielded
    every `checkpoint_iterations` iterations.
    :rtype: iterator[tuple[int|None,image]]
    """
    shape = (1,) + content.shape

    content_features = {}
    style_features = {}
    style_layers_weights = {}
    content_layer = CONTENT_LAYERS[content_layer_num]

    for i, style_layer in enumerate(STYLE_LAYERS):
        style_layers_weights[style_layer] = style_layer_weight[i]

    vgg_weights, vgg_mean_pixel = vgg.load_net(network)
    image = tf.placeholder(tf.float32 , shape=shape)
    net = vgg.net_preloaded(vgg_weights, image, pooling)

    content_pre = np.array([vgg.preprocess(content, vgg_mean_pixel)])
    style_pre = np.array([vgg.preprocess(style, vgg_mean_pixel)])

    # compute content features,style features in feedforward mode
    with tf.Session() as sess:
        content_features[content_layer] = sess.run(net[content_layer], feed_dict={image: content_pre})

        for layer in STYLE_LAYERS:
            features = sess.run(net[layer], feed_dict={image: style_pre})
            features = np.reshape(features, (-1, features.shape[3]))
            gram = np.matmul(features.T, features) / features.size
            style_features[layer] = gram

    # make stylized image using backpropogation
    if initial is None:
        noise = np.random.normal(size=shape, scale=np.std(content) * 0.1)
        initial = tf.random_normal(shape) * 0.256
    else:
        initial = np.array([vgg.preprocess(initial, vgg_mean_pixel)])
        initial = initial.astype(np.float32)
        noise = np.random.normal(size=shape, scale=np.std(content) * 0.1)
        initial = initial*(1-initial_noiseblend) + (tf.random_normal(shape)*0.256) *initial_noiseblend
    image = tf.Variable(initial)
    net = vgg.net_preloaded(vgg_weights, image, pooling)

    # content loss
    content_loss =  content_weight * 2 * tf.nn.l2_loss(
        net[content_layer] - content_features[content_layer]) /content_features[content_layer].size

    # style loss
    style_loss = 0
    for style_layer in STYLE_LAYERS:
        layer = net[style_layer]
        _, height, width, number = map(lambda i: i.value, layer.get_shape())
        size = height * width * number
        feats = tf.reshape(layer, (-1, number))
        gram = tf.matmul(tf.transpose(feats), feats) / size
        style_gram = style_features[style_layer]
        style_loss += style_weight*style_layers_weights[style_layer] * 2 * tf.nn.l2_loss(gram - style_gram) / style_gram.size

    # total variation denoising
    tv_y_size = _tensor_size(image[:, 1:, :, :])
    tv_x_size = _tensor_size(image[:, :, 1:, :])
    tv_loss = tv_weight * 2 * (
        (tf.nn.l2_loss(image[:, 1:, :, :] - image[:, :shape[1] - 1, :, :]) /
         tv_y_size) +
        (tf.nn.l2_loss(image[:, :, 1:, :] - image[:, :, :shape[2] - 1, :]) /
         tv_x_size))

    # overall loss
    loss = content_loss + style_loss + tv_loss

    # optimizer setup
    train_step = tf.train.AdamOptimizer(learning_rate, beta1, beta2, epsilon).minimize(loss)

    def print_progress():
        print('  content loss: %g\n' % content_loss.eval())
        print('    style loss: %g\n' % style_loss.eval())
        print('       tv loss: %g\n' % tv_loss.eval())
        print('    total loss: %g\n' % loss.eval())

    # optimization
    best_loss = float('inf')
    best = None
    images = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print('Optimization started...\n')
        if (print_iterations and print_iterations != 0):
            print_progress()
        for i in range(iterations):
            train_step.run()

            last_step = (i == iterations - 1)
            if last_step or (print_iterations and i % print_iterations == 0):
                print('Iteration %4d/%4d\n' % (i + 1, iterations))
                print_progress()

            if (checkpoint_iterations and i % checkpoint_iterations == 0) or last_step:
                this_loss = loss.eval()

                styled_image = np.clip(vgg.unprocess(image.eval().reshape(shape[1:]), vgg_mean_pixel), 0, 255)

                if this_loss < best_loss:
                    best_loss = this_loss
                    best = styled_image


                if preserve_colors and preserve_colors == True:
                    original_image = np.clip(content, 0, 255)

                    # Luminosity transfer steps:
                    # 1. Convert stylized RGB->grayscale accoriding to Rec.601 luma (0.299, 0.587, 0.114)
                    # 2. Convert stylized grayscale into YUV (YCbCr)
                    # 3. Convert original image into YUV (YCbCr)
                    # 4. Recombine (stylizedYUV.Y, originalYUV.U, originalYUV.V)
                    # 5. Convert recombined image from YUV back to RGB

                    # 1
                    styled_grayscale = rgb2gray(styled_image)
                    styled_grayscale_rgb = gray2rgb(styled_grayscale)

                    # 2
                    styled_grayscale_yuv = np.array(
                        Image.fromarray(styled_grayscale_rgb.astype(np.uint8)).convert('YCbCr'))

                    # 3
                    original_yuv = np.array(Image.fromarray(original_image.astype(np.uint8)).convert('YCbCr'))

                    # 4
                    w, h, _ = original_image.shape
                    combined_yuv = np.empty((w, h, 3), dtype=np.uint8)
                    combined_yuv[..., 0] = styled_grayscale_yuv[..., 0]
                    combined_yuv[..., 1] = original_yuv[..., 1]
                    combined_yuv[..., 2] = original_yuv[..., 2]

                    # 5
                    styled_image = np.array(Image.fromarray(combined_yuv, 'YCbCr').convert('RGB'))

                plt.figure(figsize = (8,8))
                plt.imshow(styled_image.astype(np.uint8))
                plt.axis('off')
                plt.show()

                images.append(styled_image.astype(np.uint8))

    return images, best

def _tensor_size(tensor):
    from operator import mul
    return reduce(mul, (d.value for d in tensor.get_shape()), 1)


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def gray2rgb(gray):
    w, h = gray.shape
    rgb = np.empty((w, h, 3), dtype=np.float32)
    rgb[:, :, 2] = rgb[:, :, 1] = rgb[:, :, 0] = gray
    return rgb


def imread(path, size = None):
    return img


def imsave(path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path, quality=95)