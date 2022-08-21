import numpy as np
import tensorflow as tf
import os


__all__ = ["img_preprocessing", "deprocess_image", "dir_init"]


def img_load(path):
    img = tf.io.decode_image(tf.io.read_file(path), channels=3)
    img = tf.image.resize(img, (512, 512))
    img = tf.expand_dims(img, axis=0)
    img = tf.keras.applications.vgg19.preprocess_input(img)
    return img


def img_preprocessing(content_path, style_path):
    content_img = img_load(content_path)
    style_img = img_load(style_path)
    gen_img = tf.Variable(img_load(content_path), dtype=tf.float32)

    return content_img, style_img, gen_img


def deprocess_image(img):
    # Util function to convert a tensor into a valid image
    img = np.squeeze(img)
    # Remove zero-center by mean pixel
    img[..., 0] += 103.939
    img[..., 1] += 116.779
    img[..., 2] += 123.68

    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def dir_init():
    save_dir = "./result"
    if os.path.isdir(save_dir):
        pass
    else:
        os.makedirs(save_dir)
