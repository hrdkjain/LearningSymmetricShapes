import numpy as np


def writeOff(file, im, gi_size=128, writeFaces=False):
  if not file.endswith('.off'):
    file += '.off'
  f = open(file, 'w+')
  R = gi_size
  C = gi_size
  if writeFaces:
    nF = (R-1)*(C-1)*2 + (R-1)*(C-1)*2 + (R-1)*2*2 + (C-1)*2*2
  else:
    nF = 0

  # Vertices
  if len(im.shape) > 2:
    if len(im.shape) == 4:
      im = np.squeeze(im, axis=0)
    f.write('OFF\n%d %d 0\n'%(np.size(im, 0)*np.size(im, 1),nF))
    for r in range(im.shape[1]):
      for c in range(im.shape[0]):
        f.write('%.5f %.5f %.5f\n'%(im[c][r][2], im[c][r][1], im[c][r][0]))
  elif len(im.shape) == 2:
    f.write('OFF\n%d %d 0\n'%(np.size(im, 0),nF))
    for c in range(im.shape[0]):
      f.write('%.5f %.5f %.5f\n'%(im[c][0], im[c][1], im[c][2]))

  # Faces
  if writeFaces:
    # slice1
    for r in range(R-1):
      for c in range(C-1):
        f.write('3 %d %d %d\n'%(r*R+c, (r+1)*R+c, (r+1)*R+c+1))
        f.write('3 %d %d %d\n'%((r+1)*R+c+1, r*R+c+1, r*R+c))

    # slice2
    nVS = R*C
    for r in range(R-1):
      for c in range(C-1):
        f.write('3 %d %d %d\n'%((r+1)*R+c+1+nVS, (r+1)*R+c+nVS, r*R+c+nVS))
        f.write('3 %d %d %d\n'%(r*R+c+nVS, r*R+c+1+nVS, (r+1)*R+c+1+nVS))

    # slice borders
    for r in range(R-1):
      c = 0
      f.write('3 %d %d %d\n'%((r+1)*R+c+nVS, (r+1)*R+c, r*R+c))
      f.write('3 %d %d %d\n'%(r*R+c, r*R+c+nVS, (r+1)*R+c+nVS))
      c = C-1
      f.write('3 %d %d %d\n' %(r*R+c, (r+1)*R+c, (r+1)*R+c+nVS))
      f.write('3 %d %d %d\n' %((r+1)*R+c+nVS, r*R+c+nVS, r*R+c))
    for c in range(C-1):
      r = 0
      f.write('3 %d %d %d\n'%(r*R+c, r*R+c+1, r*R+c+1+nVS))
      f.write('3 %d %d %d\n'%(r*R+c+1+nVS, r*R+c+nVS, r*R+c))
      r = R - 1
      f.write('3 %d %d %d\n' %(r*R+c+1+nVS, r*R+c+1, r*R+c))
      f.write('3 %d %d %d\n' %(r*R+c, r*R+c+nVS, r*R+c+1+nVS))

  f.close()