# Get keypoint data from the images and prepare for pix2pix
import os
from glob import glob
import pickle as pkl

from utils import *

inputFolder = 'data/images/v1/'
outputFolderKp = 'data/'
saveFilename = outputFolderKp + 'kp1.pickle'
outputFolderForpix2pix = 'data/outPix/'
inputToA2KeyModel = 'data/outA2K/'

os.makedirs(outputFolderForpix2pix, exist_ok=True)
os.makedirs(outputFolderKp, exist_ok=True)
os.makedirs(inputToA2KeyModel, exist_ok=True)

searchNames = inputFolder + '*' + '.jpg'
filenames = sorted(glob(searchNames))

d = []
for file in tqdm(filenames):
    img = cv2.imread(file)
    x = int(np.floor((img.shape[1] - 256) / 2))

    # Crop to a square image
    crop_img = img[0:256, x:x + 256]

    # extract the keypoints
    keypoints = get_facial_landmarks(crop_img)
    l = getKeypointFeatures(keypoints)
    unit_kp, N, tilt, mean = l[0], l[1], l[2], l[3]
    kp_mouth = unit_kp[48:68]
    store_list = [kp_mouth, N, tilt, mean, unit_kp, keypoints]
    d.append(store_list)

    # create a patch based on the tilt, mean and the size of face
    mean_x, mean_y = int(mean[0]), int(mean[1])
    size = int(N / 15)
    aspect_ratio_mouth = 1.8

    patch_img = crop_img.copy()
    # patch = np.zeros_like(patch_img[ mean_y-size: mean_y+size, mean_x-size: mean_x+size ])
    patch_img[mean_y - size: mean_y + size, mean_x - int(aspect_ratio_mouth * size):
                                            mean_x + int(aspect_ratio_mouth * size)] = 0
    cv2.imwrite(inputToA2KeyModel + os.path.splitext(os.path.basename(file))[0] + '.png', patch_img)

    drawLips(keypoints, patch_img)

    # Slap the other original image onto this
    patch_img = np.hstack((patch_img, crop_img))

    outputNamePatch = outputFolderForpix2pix + os.path.splitext(os.path.basename(file))[0] + '.png'
    cv2.imwrite(outputNamePatch, patch_img)

# save the extracted keypoints
with open(saveFilename, "wb") as output_file:
    pkl.dump(d, output_file)
