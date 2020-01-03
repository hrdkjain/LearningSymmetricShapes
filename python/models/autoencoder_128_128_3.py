import tensorflow as tf

from layers.resnet import ResidualBlock, ResidualUpsamplingBlock


def model_fn(features, labels, mode, params):
  imgs = features['encoder_input'] if isinstance(features, dict) else features

  encoder = create_encoder()
  decoder = create_decoder()

  if mode == tf.estimator.ModeKeys.PREDICT:
    encoding = encoder(imgs, training=False)
    decoding = decoder(encoding, training=False)

    return tf.estimator.EstimatorSpec(
      mode=tf.estimator.ModeKeys.PREDICT,
      predictions={"prediction": decoding}
    )

  labels, masks = labels if isinstance(labels, tuple) else (labels, None)

  if mode == tf.estimator.ModeKeys.TRAIN:

    encoding = encoder(imgs, training=True)
    decoding = decoder(encoding, training=True)

    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
      optimizer = tf.train.AdamOptimizer(learning_rate=params['lr'])

    if params['use_mask']:
      loss = tf.reduce_mean(tf.square(labels-decoding)*(masks+1.0))
    else:
      loss = tf.reduce_mean(tf.square(labels-decoding))

    return tf.estimator.EstimatorSpec(
      mode=tf.estimator.ModeKeys.TRAIN,
      loss=loss,
      train_op=optimizer.minimize(loss, tf.train.get_or_create_global_step()),
    )

  if mode == tf.estimator.ModeKeys.EVAL:

    encoding = encoder(imgs, training=False)
    decoding = decoder(encoding, training=False)

    if params['use_mask']:
      loss = tf.reduce_mean(tf.square(labels-decoding)*(masks+1.0))
    else:
      loss = tf.reduce_mean(tf.square(labels-decoding))

    return tf.estimator.EstimatorSpec(
      mode=tf.estimator.ModeKeys.EVAL,
      loss=loss,
    )


def create_encoder():
  """
  Encoder block similar to SurfNet paper. However using standard ResNet blocks for now.
  """
  return tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=2, padding='same'),
    # 64x64x16
    ResidualBlock(3, 64, 2),
    ResidualBlock(3, 64, 1),
    ResidualBlock(3, 64, 1),
    # 32x32x64
    ResidualBlock(3, 96, 2),
    ResidualBlock(3, 96, 1),
    ResidualBlock(3, 96, 1),
    # 16x16x96
    ResidualBlock(3, 128, 2),
    ResidualBlock(3, 128, 1),
    ResidualBlock(3, 128, 1),
    # 8x8x128
    ResidualBlock(3, 256, 2),
    ResidualBlock(3, 256, 1),
    ResidualBlock(3, 256, 1),
    # 4x4x256
    ResidualBlock(3, 512, 2),
    ResidualBlock(3, 512, 1),
    ResidualBlock(3, 512, 1)
    # 2x2x512
  ])


def create_decoder():
  """
  Decoder block similar to SurfNet paper.
  """
  return tf.keras.Sequential([
    ResidualUpsamplingBlock(2, 512, 2),
    ResidualBlock(3, 512, 1),
    ResidualBlock(3, 512, 1),
    # 4x4x512
    ResidualUpsamplingBlock(2, 256, 2),
    ResidualBlock(3, 256, 1),
    ResidualBlock(3, 256, 1),
    # 8x8x256
    ResidualUpsamplingBlock(2, 128, 2),
    ResidualBlock(3, 128, 1),
    ResidualBlock(3, 128, 1),
    # 16x16x128
    ResidualUpsamplingBlock(2, 96, 2),
    ResidualBlock(3, 96, 1),
    ResidualBlock(3, 96, 1),
    # 32x32x96
    ResidualUpsamplingBlock(2, 64, 2),
    ResidualBlock(3, 64, 1),
    ResidualBlock(3, 64, 1),
    # 64x64x64
    ResidualUpsamplingBlock(2, 32, 2),
    ResidualBlock(3, 32, 1),
    ResidualBlock(3, 32, 1),
    # 128x128x32
    tf.keras.layers.Conv2D(
      filters=3, kernel_size=3, strides=1,
      padding='same', activation=tf.keras.activations.sigmoid
    ),
    # 128x128x3
  ])
