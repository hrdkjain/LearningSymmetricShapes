import numpy as np


def writeOff(file, im):
  if not file.endswith('.off'):
    file += '.off'

  if len(im.shape) > 2:
    if len(im.shape) == 4:
      im = np.squeeze(im, axis=0)
    f = open(file, 'w+')
    f.write('OFF\n%d 0 0\n'%(np.size(im, 0)*np.size(im, 1)))
    for r in range(im.shape[1]):
      for c in range(im.shape[0]):
        f.write('%.5f %.5f %.5f\n'%(im[c][r][2], im[c][r][1], im[c][r][0]))
    f.close()
  elif len(im.shape) == 2:
    f = open(file, 'w+')
    f.write('OFF\n%d 0 0\n'%(np.size(im, 0)))
    for c in range(im.shape[0]):
      f.write('%.5f %.5f %.5f\n'%(im[c][0], im[c][1], im[c][2]))
    f.close()
