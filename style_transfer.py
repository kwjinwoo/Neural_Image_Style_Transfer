import tensorflow as tf
from tensorflow import keras
from configs import cfg
from loss import total_loss
from utils import *
import cv2
import argparse


parser = argparse.ArgumentParser(description='Neural Style Transfer script')
parser.add_argument('--style_img', required=True, type=str, help="style image path")
parser.add_argument('--content_img', required=True, type=str, help="content image path")

args = parser.parse_args()


def get_extractor():
    vgg = tf.keras.applications.VGG19(weights="imagenet", include_top=False)
    for layer in vgg.layers:
        layer.trainable = False
    names = cfg.style_conv_names + [cfg.content_conv_name]
    outputs = dict([(layer.name, layer.output) for layer in vgg.layers if layer.name in names])
    return keras.models.Model(vgg.input, outputs)


@tf.function
def compute_loss_grad(content_img, style_img, gen_img, feature_extractor):
    with tf.GradientTape() as tape:
        loss = total_loss(feature_extractor, style_img, content_img, gen_img)
    grads = tape.gradient(loss, gen_img)
    return loss, grads


if __name__ == "__main__":
    content_path = args.content_img
    style_path = args.style_img

    dir_init()

    content_img, style_img, gen_img = img_preprocessing(content_path, style_path)
    feature_extractor = get_extractor()

    optimizer = keras.optimizers.Adam(learning_rate=1e1)

    for i in range(1, cfg.iteration + 1):
        loss, grads = compute_loss_grad(content_img, style_img, gen_img, feature_extractor)
        optimizer.apply_gradients([(grads, gen_img)])

        if i % cfg.inter_save == 0:
            print("Iteration %d: loss=%.2f" % (i, loss))
            img = deprocess_image(gen_img.numpy())
            save_path = "./result/result_iter_%d.png" % i
            cv2.imwrite(save_path, img)
