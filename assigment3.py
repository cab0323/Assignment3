from calendar import c
from pydoc import locate
from re import template
from telnetlib import SSPI_LOGON
import cv2
from cv2 import CV_8U
from cv2 import medianBlur
from cv2 import PyRotationWarper
from matplotlib import cm
import numpy as np
import pywt
import matplotlib.pyplot as plt

#1.1
imgPath = r'cone.jpg'
img = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
imgResize = cv2.resize(img, (0,0), fx = 0.5, fy = 0.5)
cv2.imshow('Image',imgResize)
cv2.waitKey(0)
cv2.destroyAllWindows()

#1.2
templatePath = r'template.jpg'
template = cv2.imread(templatePath, cv2.IMREAD_GRAYSCALE)
cv2.imshow('Template', template)
cv2.waitKey(0)
cv2.destroyAllWindows()

#1.3
#median filter
median = medianBlur(template, 5)
cv2.imshow('Median Filter', median)
cv2.waitKey(0)
cv2.destroyAllWindows()


#1.4 - 1.6
#template matching
h, w = template.shape
methods = [cv2.TM_CCORR_NORMED, cv2.TM_CCORR,cv2.TM_SQDIFF]
for method in methods:
    imgCpy = img.copy()
    result = cv2.matchTemplate(imgCpy, template, method)

    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)

    if method in [cv2.TM_SQDIFF]:
        location = minLoc
    else:
        location = maxLoc

    bottomRight = (location[0] + w, location[1] + h)
    cv2.rectangle(imgCpy, location, bottomRight, 255, 5)

    cv2.imshow('template match', imgCpy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#QUESTION 2

#2.1
wavelet = pywt.Wavelet('bior1.3')


#2.2
[phiD, psiD, phiR, psiR, x] = wavelet.wavefun()
plt.subplot(1, 2, 1)
plt.plot(x, phiD)
plt.title("PHI")
#plt.xticks([]), plt.yticks([])
plt.subplot(1, 2, 2)
plt.plot(x, psiD)
plt.title("PSI")
#plt.xticks([]), plt.yticks([])
#plt.show()   #showing the plots



#2.4
ssPath = r'skyscrapers.jpg'
ss = cv2.imread(ssPath, cv2.IMREAD_GRAYSCALE)
cv2.imshow('Skyscrapers Image', ss)
cv2.waitKey(0)
cv2.destroyAllWindows()


#2.5
coeffs = pywt.dwt2(ss, wavelet)

#2.6
A, (detailH, detailV, detailD) = coeffs
plt.subplot(2, 2, 1)
plt.title("Approximation")
plt.imshow(A, cmap = 'gray', interpolation='nearest')
plt.subplot(2, 2, 2)
plt.title("Horizontal Detail")
plt.imshow(detailH, cmap = 'gray', interpolation='nearest')
plt.subplot(2, 2, 3)
plt.title("Vertical Detail")
plt.imshow(detailV, cmap= 'gray', interpolation= 'nearest')
plt.subplot(2, 2, 4)
plt.title("Diagonal Detail")
plt.imshow(detailD, cmap= 'gray', interpolation='nearest')
plt.show()


#2.7
wavelet2 = pywt.Wavelet('db20')

#2.8
phi, psi, x = wavelet2.wavefun()

plt.subplot(1, 2, 1)
plt.title("PHI")
plt.plot(x, phi)
plt.subplot(1, 2, 2)
plt.title("PSI")
plt.plot(x, psi)
plt.show()


#2.9
ssArr = np.array(ss)
coeffs = pywt.dwt2(ssArr, wavelet2)

#2.10
Appr, (hD, vD, dD) = coeffs
plt.subplot(2, 2, 1)
plt.title("Approximation")
plt.imshow(Appr, cmap= 'gray', interpolation= 'nearest')
plt.subplot(2, 2, 2)
plt.imshow(hD, cmap= 'gray', interpolation='nearest')
plt.title("Horizontal Detail")
plt.subplot(2, 2, 3)
plt.title("Vertical Detail")
plt.imshow(vD, cmap= 'gray', interpolation= 'nearest')
plt.subplot(2, 2, 4)
plt.title("Diagonal Detail")
plt.imshow(dD, cmap= 'gray', interpolation= 'nearest')
plt.show()


#2.12
ss_poly = ss.copy()
#drawing the lines of the polynomial
cv2.line(ss_poly, (350, 388), (260,388), (0,0,0), 1)#bottom
cv2.line(ss_poly, (260,388), (220, 338), (0,0,0), 1)
cv2.line(ss_poly, (220,338), (220,278), (0,0,0), 1)#left vertical
cv2.line(ss_poly, (220,278), (260, 228), (0,0,0), 1)
cv2.line(ss_poly, (260, 228), (350,228), (0,0,0), 1)#top
cv2.line(ss_poly, (350,228), (390, 278), (0,0,0), 1)
cv2.line(ss_poly, (390,278), (390,338), (0,0,0), 1)#right vertical
cv2.line(ss_poly, (390,338), (350, 388), (0,0,0), 1)

plt.title("ss_poly")
plt.imshow(ss_poly, cmap= 'gray')
plt.show()

#2.13

coeffs = pywt.dwt2(ss_poly, wavelet)
A, (hD, vD, dD) = coeffs

plt.subplot(2, 2, 1)
plt.title("Approximation")
plt.imshow(A, cmap= 'gray', interpolation= 'nearest')
plt.subplot(2, 2, 2)
plt.title("Horizontal Detail")
plt.imshow(hD, cmap= 'gray', interpolation= 'nearest')
plt.subplot(2, 2, 3)
plt.title("Vertical Detail")
plt.imshow(vD, cmap= 'gray', interpolation= 'nearest')
plt.subplot(2, 2, 4)
plt.title("Diagonal Detail")
plt.imshow(dD, cmap= 'gray', interpolation= 'nearest')
plt.show()