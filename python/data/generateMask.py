import os

import numpy as np
from skimage import img_as_float, io
from tqdm import tqdm


def generateMask(inFlPath, outFlPath):
  if not os.path.exists(outFlPath):
    os.makedirs(outFlPath)

  # List all the GI images of the fl path
  flGIList = os.listdir(inFlPath)
  flGIList = list(filter(lambda x: x.endswith('.png'), flGIList))
  flGIList = list(filter(lambda x: '_flatGI' in x, flGIList))

  # Find its corresponding nGI
  for gi in tqdm(flGIList):
    ngi = gi.replace('_flatGI', '_nflatGI')
    ngiPath = os.path.join(inFlPath, ngi)
    mgi = ngi.replace('_nflatGI', '_mflatGI')
    mgiOutPath = os.path.join(outFlPath, mgi)
    if os.path.exists(ngiPath):
      if os.path.exists(mgiOutPath):
        continue
      scalarMask(ngiPath, mgiOutPath)
    else:
      print('nGI: %s not found\n'%ngiPath)

  print('\nMask files written to: %s'%outFlPath)


def scalarMask(ngiPath, mgiOutPath, windowSize=1):
  ngi = io.imread(ngiPath)
  ngi = img_as_float(ngi)
  mgiOut = np.empty([ngi.shape[0], ngi.shape[1]], dtype=float)
  for r in range(ngi.shape[0]):
    for c in range(ngi.shape[1]):
      v0 = ngi[r, c, :]
      angleSum = 0
      pixelCount = 0
      for rr in range(r-windowSize, r+windowSize+1):
        for cc in range(c-windowSize, c+windowSize+1):

          if rr < 0 or rr >= ngi.shape[0] or cc < 0 or cc >= ngi.shape[1]:
            # skip out of image content
            continue
          elif rr == r or cc == c:
            # skip self
            continue
          pixelCount = pixelCount+1;
          v1 = ngi[rr, cc, :]
          if np.dot(v0, v1) == 0:
            angle = 1.57
          elif (np.array_equal(v0, v1)):
            angle = 0
          else:
            angle = np.arccos(myDot(v0, v1)/(np.linalg.norm(v0)*np.linalg.norm(v1)))  # *180/3.14

          if angle < 0:
            print(angle)
          elif np.isnan(angle):
            continue
          angleSum += abs(angle)
      mgiOut[r, c] = angleSum/pixelCount

  # scale to [0,1]
  mgiOut = mgiOut-mgiOut.min()
  mgiOut = mgiOut/mgiOut.max()
  io.imsave(mgiOutPath, img_as_float(mgiOut))


def myDot(v0, v1):
  return v0[0]*v1[0]+v0[1]*v1[1]+v0[2]*v1[2]


if __name__ == "__main__":
  inFlPath = '../../Example/GI'
  outFlPath = '../../Example/msk'
  generateMask(inFlPath, outFlPath)
