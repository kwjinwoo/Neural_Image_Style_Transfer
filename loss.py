import tensorflow as tf
from configs import cfg


def content_loss(content_feature, gen_feature):
    return tf.reduce_sum(tf.square(content_feature - gen_feature)) / 2.


def gram_matrix(feature_map):
    feature_map = tf.transpose(feature_map, (2, 0, 1))
    feature_map = tf.reshape(feature_map, (tf.shape(feature_map)[0], -1))
    gram = tf.matmul(feature_map, tf.transpose(feature_map))
    return gram


def style_loss(style_feature, gen_feature):
    feature_map_size = tf.cast(tf.shape(style_feature)[0] * tf.shape(style_feature)[1], dtype=tf.float32)
    num_filters = tf.cast(tf.shape(style_feature)[-1], dtype=tf.float32)
    scale_factor = 4. * (feature_map_size ** 2) * (num_filters ** 2)

    style_gram = gram_matrix(style_feature)
    gen_gram = gram_matrix(gen_feature)

    return tf.reduce_sum(tf.square(style_gram - gen_gram)) / scale_factor


def total_loss(feature_extractor, style_img, content_img, gen_img):
    inputs = tf.concat([style_img, content_img, gen_img], axis=0)
    features = feature_extractor(inputs)

    # init loss
    loss = tf.zeros(shape=())

    # content
    content_features = features[cfg.content_conv_name]
    content_feature_map = content_features[1, ...]
    gen_content_feature_amp = content_features[2, ...]
    loss += cfg.alpha * content_loss(content_feature_map, gen_content_feature_amp)

    # style
    sl = tf.zeros(shape=())
    for name in cfg.style_conv_names:
        style_features = features[name]
        style_feature_map = style_features[0, ...]
        gen_style_feature_map = style_features[2, ...]
        sl += cfg.weighting_factor * style_loss(style_feature_map, gen_style_feature_map)
    loss += cfg.beta * sl

    return loss


if __name__ == "__main__":
    temp1 = tf.random.normal(shape=(1, 32, 32, 512))
    temp2 = tf.random.normal(shape=(32, 32, 512))
    print(gram_matrix(temp2).shape)
