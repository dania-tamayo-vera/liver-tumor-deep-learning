import SimpleITK as sitk
import cv2
import numpy as np
import os

from matplotlib import pyplot as plt

center = [-10, 20, 50, 80]
width = [80, 100, 140, 180]


def saveNii(img, seg_liver, sort):
    for n in range(0, seg_liver.shape[0], 4):
        sum_array = sum(sum((seg_liver[n, :, :])))
        if sum_array != 0:
            for i in range(len(center)):
                for j in range(len(width)):
                    dir1 = r"E:/LITS/liver1/%d-%d/%s/" % (center[i], width[j], sort)
                    names = os.listdir(dir1)
                    p = int(len(names) / 2)
                    silce = img[n, :, :]
                    min = (2 * center[i] - width[j]) / 2.0 + 0.5
                    max = (2 * center[i] + width[j]) / 2.0 + 0.5
                    dFactor = 255.0 / (max - min)
                    silce = silce - min
                    silce = np.trunc(silce * dFactor)
                    silce[silce < 0.0] = 0
                    silce[silce > 255.0] = 255
                    cv2.imwrite("E:/LITS/liver1/%d-%d/%s/%d_body.png" % (center[i], width[j], sort, p), silce)
                    cv2.imwrite("E:/LITS/liver1/%d-%d/%s/%d_liver.png" % (center[i], width[j], sort, p),
                                seg_liver[n, :, :] * 255)
                    p = p + 1


for m in range:
    itk_img = sitk.ReadImage('D:/LITS/ct/volume-%d.nii' % m)
    itk_imgmask = sitk.ReadImage('D:/LITS/seg/segmentation-%d.nii' % m)
img = sitk.GetArrayFromImage(itk_img)
imgmask = sitk.GetArrayFromImage(itk_imgmask)
seg_liver = img.copy()
seg_liver[seg_liver > 0] = 1

seg_tumorimage = img.copy()
seg_tumorimage[img == 1] = 0
seg_tumorimage[img == 2] = 1
if m < 120:
    sort = "train"
else:
    sort = "test"
sort = "train"
saveNii(img, seg_liver, sort)
