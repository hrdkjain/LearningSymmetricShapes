import os

import numpy as np
import skimage.io
from natsort import natsorted
from tqdm import tqdm

from data.utils import writeOff


def save_params(params):
  if not os.path.exists(os.path.join(params['model_dir'])):
    os.makedirs(os.path.join(params['model_dir']))

  with open(os.path.join(params['model_dir'], 'parameters.txt'), 'w+')  as f:
    for key in params.keys():
      value = params[key]
      if (isinstance(value, str) or isinstance(value, int) or isinstance(value, float) or isinstance(value, bool)):
        f.write('%s: %s\n'%(key, str(value)))
      elif (isinstance(value, list)):
        f.write('%s:\n'%key)
        for v in value:
          f.write(v+'\n')
  f.close()


def load_pairs(path, params, instance):
  # main loader function
  if 'test_on_real' in params.keys() and params['test_on_real']:
    return load_realRGB(path)

  return load_GI(path, params, instance)


def load_GI(path, params, instance):
  # prepare list of valid rgb files and then generate pair wise np array and path list
  rgb_list = []
  gi_list = []
  mask_list = []

  gi_path = os.path.join(path, instance)
  rgb_path = os.path.join(path, instance+'_rgb')
  mask_path = os.path.join(path, instance+'_msk')

  rgb_files = natsorted(os.listdir(rgb_path))
  gi_files = natsorted(os.listdir(gi_path))
  if params['use_mask']:
    msk_files = natsorted(os.listdir(mask_path))

  # prepare rgb list
  print('Listing %s files'%instance)
  for rgb_file in tqdm(rgb_files):
    if rgb_file.endswith('.png'):
      f = rgb_file.strip('.png')
      f = f.split('_')
      fView = f[-1]
      file = f[0]+'_'+f[1]
      gi_file = file+'_'+params['parameterization_suffix']+'_'+str(params['gi_size'])+'_flatGI.png'
      if params['gi_channels'] == 3:
        if fView in params['SelectedViews'] and gi_file in gi_files:
          # add file path list
          rgb_list.append(os.path.join(rgb_path, rgb_file))
          gi_list.append((os.path.join(gi_path, gi_file),))
      elif params['gi_channels'] == 6:
        ngi_file = file+'_'+params['parameterization_suffix']+'_'+str(params['gi_size'])+'_nflatGI.png'
        if fView in params['SelectedViews'] and gi_file in gi_files and ngi_file in gi_files:
          # add file path list
          rgb_list.append(os.path.join(rgb_path, rgb_file))
          gi_list.append((os.path.join(gi_path, gi_file), os.path.join(gi_path, ngi_file)))

      if params['use_mask']:
        mask_file = file+'_'+params['parameterization_suffix']+'_'+str(params['gi_size'])+'_m'+'flatGI.png'
        if fView in params['SelectedViews'] and mask_file in msk_files:
          # add file path list
          mask_list.append(os.path.join(mask_path, mask_file))

  print('Listed %d rgb files'%len(rgb_list))
  print('Listed %d gi files'%len(gi_list))
  print('Listed %d mask files'%len(mask_list))

  # assertions
  assert (len(rgb_list) == len(gi_list)), 'list size mismatch'
  if params['use_mask']:
    assert (len(rgb_list) == len(mask_list)), 'list size mismatch'

  meanFile = os.path.join(params['model_dir'], 'meanShape.off')
  if params['generateMeanShape'] and instance == 'trn' and not os.path.exists(meanFile):
    unique_gi_list = list(set(gi_list))
    unique_gi_np = np.empty([len(unique_gi_list), params['gi_size'], params['gi_size'], params['gi_channels']])
    print('Loading unique gi files')
    for i in tqdm(range(len(unique_gi_list))):
      im = np.array(skimage.io.imread(unique_gi_list[i][0]))
      im = skimage.img_as_float32(im)
      unique_gi_np[i, :, :, 0:3] = skimage.transform.resize(im, [params['gi_size'], params['gi_size']])
    mean_unique_gi = np.mean(unique_gi_np, axis=0)
    if not os.path.exists(params['model_dir']):
      os.makedirs(params['model_dir'])
    writeOff(meanFile, mean_unique_gi)
    skimage.io.imsave(os.path.join(params['model_dir'], 'meanShape.png'), mean_unique_gi[:, :, 0:3])
    print('Mean shape saved %s'%meanFile)

  if params['loadFiles']:
    rgb_np = np.empty([len(rgb_list), params['rgb_size'], params['rgb_size'], params['rgb_channels']])
    gi_np = np.empty([len(gi_list), params['gi_size'], params['gi_size'], params['gi_channels']])
    print('Loading gi files')
    for i in tqdm(range(len(gi_list))):
      im = np.array(skimage.io.imread(gi_list[i][0]))
      im = skimage.img_as_float32(im)
      gi_np[i, :, :, 0:3] = im

      if params['gi_channels'] == 6 and len(gi_file) == 2:
        im2 = np.array(skimage.io.imread(gi_list[i][1]))
        im2 = skimage.img_as_float32(im2)
        gi_np[i, :, :, 3:6] = im2

    print('Loading rgb files')
    for i in tqdm(range(len(rgb_list))):
      img = np.array(skimage.io.imread(rgb_list[i]))
      img = skimage.img_as_float32(img)
      rgb_np[i, :, :, 0:params['rgb_channels']] = img
    return {'rgb_np': rgb_np, 'gi_np': gi_np, 'rgb_list': rgb_list, 'gi_list': gi_list}

  return {'rgb_list': rgb_list, 'gi_list': gi_list, 'mask_list': mask_list}


def load_realRGB(path, instance):
  # prepare list of valid rgb files and then generate pair wise np array and path list
  rgb_list = []
  gi_list = []
  mask_list = []

  rgb_path = os.path.join(path, instance)
  rgb_list = natsorted(os.listdir(rgb_path))
  rgb_list = list(filter(lambda x: x.endswith('.png'), rgb_list))
  rgb_list = [os.path.join(rgb_path, rgb_file) for rgb_file in rgb_list]
  print('Listed %d rgb files'%len(rgb_list))
  return {'rgb_list': rgb_list, 'gi_list': gi_list, 'mask_list': mask_list}


def load_geoimgs(path, params, instance):
  path = os.path.join(path, instance)

  f_out = np.array([])
  fileList = []
  files = sorted(os.listdir(path))
  print("%s %s %s"%('loading', instance, 'geoimgs'))
  for f1 in tqdm(files):
    if '_flatGI.png' in f1:
      # load flatGI image
      img1 = np.array(skimage.io.imread(os.path.join(path, f1)))
      img1 = skimage.img_as_float32(img1)
      img1 = img1[np.newaxis, :, :, :]

      # check for nChannels
      if params.nchannels == 6:
        # load the _nflatGI
        f2 = f1.replace('_flatGI.png', '_nflatGI.png')
        if f2 in files:
          img2 = np.array(skimage.io.imread(os.path.join(path, f2)))
          img2 = skimage.img_as_float32(img2)
          img2 = img2[np.newaxis, :, :, :]
          # concatenate img1 and img2
          im = np.concatenate((img1, img2), axis=3)
        else:
          # this is a trouble, in this case ignore the _flatGI as well
          continue
      elif params.nchannels == 3:
        # dont load anything else and assign flatGI to im
        im = img1
      else:
        continue

      # all the assignments were successful to append it to the list of files
      f_out = np.concatenate((f_out, im), axis=0) if bool(f_out.size) else im

      f = f1.split('_')
      fileList.append(f[0]+'_'+f[1])
  print("%s %s %s"%('loaded', str(len(fileList)), 'geo images'))
  return f_out, fileList


def load_rgbimgs(path, params, instance):
  path = os.path.join(path, instance+'_rgb')
  f_out = np.array([])
  fileList = []
  files = sorted(os.listdir(path))

  print("%s %s %s"%('loading', instance, 'rgbimgs'))
  for f in tqdm(files):
    if f.endswith('.png'):
      for view in params.SelectedViews:
        if view in f:
          img = np.array(skimage.io.imread(os.path.join(path, f)), dtype=np.uint8)
          img = skimage.img_as_float32(img)
          if img.shape[2] != 3:
            img = img[:, :, np.newaxis]
            img = np.concatenate((img, img, img), axis=2)
            print('%s is a gray scale image'%f)
          img = img[np.newaxis, :, :, :]
          f_out = np.concatenate((f_out, img), axis=0) if bool(f_out.size) else img

          f1 = f.split('_')
          fileList.append(f1[0]+'_'+f1[1])
  print("%s %s %s"%('loaded', str(len(fileList)), 'rgb images'))
  return f_out, fileList
