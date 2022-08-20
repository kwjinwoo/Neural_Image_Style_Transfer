import tensorflow as tf
from configs import cfg


def content_loss(content_feature, gen_feature):
    return tf.reduce_sum(tf.square(content_feature, gen_feature))


def gram_matrix(feature_map):
    feature_map = tf.transpose(feature_map, perm=[2, 0, 1])
    feature_map = tf.reshape(feature_map, shape=(tf.shape(feature_map)[0], -1))
    return tf.matmul(feature_map, feature_map, transpose_b=True)


def style_loss(style_feature, gen_feature):
    feature_map_size = tf.shape(style_feature)[1] * tf.shape(style_feature)[2]
    num_filters = tf.shape(style_feature)[0]
    scale_factor = 1 / (4 * feature_map_size * num_filters)

    style_gram = gram_matrix(style_feature)
    gen_feature = gram_matrix(gen_feature)

    return scale_factor * tf.reduce_sum(tf.square(style_gram - gen_feature))


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
