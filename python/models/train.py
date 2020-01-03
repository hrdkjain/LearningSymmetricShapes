import tensorflow as tf
from tqdm import tqdm

from data.dataset import dataset
from data.loaders import load_pairs


def train(model_fn, params):
  def trn_input_fn():
    trn_set = load_pairs(params['data_dir'], params, 'trn')
    rgb_imgs, geo_imgs, msks = trn_set['rgb_list'], trn_set['gi_list'], trn_set['mask_list']
    msks = msks if params['use_mask'] else None

    # Converting list of tuple to list of string
    geo_imgs = [x[0] for x in geo_imgs]

    return dataset(
      rgb_imgs, geo_imgs, msks,
      params['batch_size'],
      repeat=params['epochs_between_evals'],
      shuffle=True,
      dim_imgs=params['rgb_size'],
      dim_lbls=params['gi_size']
    )

  def val_input_fn():
    val_set = load_pairs(params['data_dir'], params, 'val')
    rgb_imgs, geo_imgs, msks = val_set['rgb_list'], val_set['gi_list'], val_set['mask_list']
    msks = msks if params['use_mask'] else None

    # Converting list of tuple to list of string
    geo_imgs = [x[0] for x in geo_imgs]

    return dataset(
      rgb_imgs, geo_imgs, msks,
      params['batch_size'],
      repeat=True,
      shuffle=True,
      dim_imgs=params['rgb_size'],
      dim_lbls=params['gi_size']
    )

  session_config = tf.ConfigProto()
  session_config.gpu_options.allow_growth = True

  runconfig = tf.estimator.RunConfig(
    model_dir=params['model_dir'],
    tf_random_seed=None,
    save_summary_steps=100,
    save_checkpoints_steps=params['instances_between_checkpoints'],
    session_config=session_config,
    keep_checkpoint_max=None,
    log_step_count_steps=100
  )

  network = tf.estimator.Estimator(
    model_fn=model_fn,
    params=params,
    config=runconfig
  )

  for _ in tqdm(range(params['epochs']//params['epochs_between_evals'])):
    network.train(input_fn=trn_input_fn)
    results = network.evaluate(input_fn=val_input_fn)
    print(results)
