import tensorflow as tf

from data.dataset import dataset
from data.loaders import load_pairs


def predict(model_fn, params, checkpoint_path=None):
  def tst_input_fn():
    tst_set = load_pairs(params['data_dir'], params, 'tst')
    rgb_imgs, _, _ = tst_set['rgb_list'], tst_set['gi_list'], tst_set['mask_list']

    return dataset(
      rgb_imgs, None, None,
      params['batch_size'],
      repeat=True,
      shuffle=False,
      dim_imgs=params['rgb_size'],
      dim_lbls=params['gi_size']
    )

  network = tf.estimator.Estimator(
    model_dir=params['model_dir'],
    model_fn=model_fn,
    params=params,
  )

  checkpoint_path = network.latest_checkpoint() if checkpoint_path is None else checkpoint_path
  prediction = network.predict(input_fn=tst_input_fn, checkpoint_path=checkpoint_path)

  tst_set = load_pairs(params['data_dir'], params, 'tst')
  return prediction, tst_set['rgb_list']
