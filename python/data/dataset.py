import tensorflow as tf


def _parse_fn_img(img, dim_img=128, dim_lbl=128):
  img = tf.image.decode_png(tf.read_file(img), channels=3)
  img = tf.cast(img, tf.float32)*(1./255)
  img = tf.image.resize_images(img, size=[dim_img, dim_img])

  return {'encoder_input': img}


def _parse_fn_lbl(img, lbl, dim_img=128, dim_lbl=128):
  img = tf.image.decode_png(tf.read_file(img), channels=3)
  img = tf.cast(img, tf.float32)*(1./255)
  img = tf.image.resize_images(img, size=[dim_img, dim_img])

  lbl = tf.image.decode_png(tf.read_file(lbl), channels=3)
  lbl = tf.cast(lbl, tf.float32)*(1./255)
  lbl = tf.image.resize_images(lbl, size=[dim_lbl, dim_lbl])

  return ({'encoder_input': img}, lbl)


def _parse_fn_msk(img, lbl, dim_img=128, dim_lbl=128):
  lbl, msk = lbl

  img = tf.image.decode_png(tf.read_file(img), channels=3)
  img = tf.cast(img, tf.float32)*(1./255)
  img = tf.image.resize_images(img, size=[dim_img, dim_img])

  lbl = tf.image.decode_png(tf.read_file(lbl), channels=3)
  lbl = tf.cast(lbl, tf.float32)*(1./255)
  lbl = tf.image.resize_images(lbl, size=[dim_lbl, dim_lbl])

  msk = tf.image.decode_png(tf.read_file(msk), channels=1)
  msk = tf.cast(msk, tf.float32)*(1./255)
  msk = tf.image.resize_images(msk, size=[dim_lbl, dim_lbl])

  return ({'encoder_input': img}, (lbl, msk))


def dataset(
    imgs,
    lbls=None,
    mask=None,
    batch_size=128,
    shuffle=True,
    repeat=None,  # positive number or None or False
    buffer_size=1000,
    num_parallel_calls=4,
    dim_imgs=128,
    dim_lbls=128
):
  imgs = tf.constant(imgs)

  if lbls is None:
    dataset = tf.data.Dataset.from_tensor_slices(imgs)
    dataset = dataset.shuffle(buffer_size=buffer_size) if shuffle else dataset
    dataset = dataset.map(lambda x: _parse_fn_img(x, dim_imgs, dim_lbls),
                          num_parallel_calls=num_parallel_calls)
  elif mask is None:
    lbls = tf.constant(lbls)
    dataset = tf.data.Dataset.from_tensor_slices((imgs, lbls))
    dataset = dataset.shuffle(buffer_size=buffer_size) if shuffle else dataset
    dataset = dataset.map(lambda x, y: _parse_fn_lbl(x, y, dim_imgs, dim_lbls),
                          num_parallel_calls=num_parallel_calls)
  else:
    lbls = tf.constant(lbls)
    mask = tf.constant(mask)
    dataset = tf.data.Dataset.from_tensor_slices((imgs, (lbls, mask)))
    dataset = dataset.shuffle(buffer_size=buffer_size) if shuffle else dataset
    dataset = dataset.map(lambda x, y: _parse_fn_msk(x, y, dim_imgs, dim_lbls),
                          num_parallel_calls=num_parallel_calls)

  dataset = dataset.batch(batch_size)
  dataset = dataset.repeat(repeat) if repeat or repeat is None else dataset
  dataset = dataset.prefetch(1)

  return dataset
