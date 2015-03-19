import cv2
import optparse
import numpy as np
import math
from os import listdir
from os.path import isfile, isdir, join

# http://www.pyimagesearch.com/2015/03/09/capturing-mouse-click-events-with-python-and-opencv/
# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []
cropping = False
def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, cropping
    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True

    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        refPt.append((x, y))
        cropping = False
        
        # draw a refPtangle around the region of interest
        cv2.rectangle(rimg, refPt[0], refPt[1], (0, 255, 0), 1)
        cv2.imshow("w1", rimg)


parser = optparse.OptionParser()
parser.add_option('-d', '--dir', dest='dir')
parser.add_option('-p', '--prefix', dest='prefix')
(options, args) = parser.parse_args()
print options
print options.dir
print options.prefix
SCALE = 0.3
for fpath in listdir(options.dir):
  if fpath[0] != 'L' or fpath[-1] != 'g':
    continue
  fpp = join(options.dir, fpath)
  img = cv2.imread(fpp, 1)
  rows = len(img)
  cols = len(img[0])
  rimg = cv2.resize(img, (0,0), fx = SCALE, fy = SCALE)
  cv2.namedWindow('w1', cv2.CV_WINDOW_AUTOSIZE)
  cv2.setMouseCallback('w1', click_and_crop)
  cv2.imshow('w1',rimg)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  fout = open(join(options.dir, fpath.split('.')[0] + '.txt'), 'w+')
  refPt[0] = (int(refPt[0][0]/SCALE), int(refPt[0][1]/SCALE))
  refPt[1] = (int(refPt[1][0]/SCALE), int(refPt[1][1]/SCALE))
  
  fout.write('{},{},{},{},{},{},{},{},'.format(refPt[0][0], refPt[0][1], refPt[0][0], refPt[1][1], refPt[1][0], refPt[1][1], refPt[1][0], refPt[0][1]))
  fout.close()
  



  
   





