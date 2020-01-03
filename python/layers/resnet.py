import tensorflow as tf


def padding(x, kernel_size):
  """ Pads input tensor with pad size kernel_size - 1.

  Args:
    x: Tensor [batch_size, height, width, channels]
    kernel_size: Integer
      The kernel used in conv and max_pool layers.

  Returns:
    x: Tensor [batch_size, height+pad_s, width+pad_e, channels]
      Padded input tensor.
  """
  pad = kernel_size-1
  pad_s = pad//2
  pad_e = pad-pad_s
  return tf.pad(x, [[0, 0], [pad_s, pad_e], [pad_s, pad_e], [0, 0]])


class ResidualBlock(tf.keras.layers.Layer):

  def __init__(self, kernel_size, filters, strides):
    super(ResidualBlock, self).__init__()

    self.kernel_size = kernel_size
    self.filters = filters
    self.strides = strides

    self.bn_i = tf.keras.layers.BatchNormalization()
    self.bn_o = tf.keras.layers.BatchNormalization()

    if self.strides > 1:
      self.conv_i = tf.keras.layers.Conv2D(filters, kernel_size, strides, padding='valid', use_bias=False)
      self.conv_p = tf.keras.layers.Conv2D(filters, 1, strides, padding='valid', use_bias=False)
    else:
      self.conv_i = tf.keras.layers.Conv2D(filters, kernel_size, strides, padding='same', use_bias=False)

    self.conv_o = tf.keras.layers.Conv2D(filters, kernel_size, 1, padding='same', use_bias=False)

  def call(self, input_tensor, training=False):

    x = self.bn_i(input_tensor)
    x = tf.nn.relu(x)

    # If we do a projection than after bn and relu since it is a convolution.
    if self.strides > 1:
      input_tensor = self.conv_p(x)
      x = padding(x, self.kernel_size)

    x = self.conv_i(x)

    x = self.bn_o(x)
    x = tf.nn.relu(x)

    x = self.conv_o(x)

    return x+input_tensor

  def compute_output_shape(self, input_shape):

    output_shape = [input_shape[0], input_shape[1], input_shape[2], self.filters]

    if self.strides > 1:
      output_shape[1] = (input_shape[1]-self.kernel_size)//self.strides+1
      output_shape[2] = (input_shape[2]-self.kernel_size)//self.strides+1

    return output_shape


class ResidualUpsamplingBlock(tf.keras.layers.Layer):

  def __init__(self, kernel_size, filters, strides):
    super(ResidualUpsamplingBlock, self).__init__()

    self.kernel_size = kernel_size
    self.filters = filters
    self.strides = strides

    self.bn_i = tf.keras.layers.BatchNormalization()
    self.bn_o = tf.keras.layers.BatchNormalization()

    self.conv_i = tf.keras.layers.Conv2DTranspose(filters, kernel_size, strides, padding='valid', use_bias=False)
    self.conv_p = tf.keras.layers.Conv2DTranspose(filters, 1, strides, padding='valid', use_bias=False)
    self.conv_o = tf.keras.layers.Conv2D(filters, kernel_size, 1, padding='same', use_bias=False)

  def call(self, input_tensor, training=False):
    x = self.bn_i(input_tensor)
    x = tf.nn.relu(x)

    # We do the projection after bn and relu since it is a convolution.
    input_tensor = self.conv_p(x)

    x = self.conv_i(x)

    x = self.bn_o(x)
    x = tf.nn.relu(x)

    x = self.conv_o(x)

    return x+input_tensor

  def compute_output_shape(self, input_shape):
    output_shape = [input_shape[0], input_shape[1], input_shape[2], self.filters]

    if self.strides > 1:
      output_shape[1] = (input_shape[1]-1)*self.strides+self.kernel_size
      output_shape[2] = (input_shape[2]-1)*self.strides+self.kernel_size

    return output_shape
