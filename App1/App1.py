import cv2
import numpy as np
import matplotlib.pyplot as plt

# SECTION 1

img = cv2.imread('im_8.jpg')
img_gray = cv2.imread('im_8.jpg', 0)  # Flag = 0 => Read the image as grayscale

B, G, R = cv2.split(img)  # Splitting the BGR image in 3 channels

imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
H, S, V = cv2.split(imgHSV)  # Splitting the HSV image in 3 channels

imgYCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
Y, Cr, Cb = cv2.split(imgYCrCb)  # Splitting the YCrCb image in 3 channels

plt.figure(1)
plt.subplot(221), plt.imshow(img_gray, cmap='gray'), plt.title("Gray")
plt.subplot(222), plt.imshow(B, cmap='gray'), plt.title("B")
plt.subplot(223), plt.imshow(G, cmap='gray'), plt.title("G")
plt.subplot(224), plt.imshow(R, cmap='gray'), plt.title("R")

plt.figure(2)
plt.subplot(221), plt.imshow(img_gray, cmap='gray'), plt.title("Gray")
plt.subplot(222), plt.imshow(H, cmap='gray'), plt.title("H")
plt.subplot(223), plt.imshow(S, cmap='gray'), plt.title("S")
plt.subplot(224), plt.imshow(V, cmap='gray'), plt.title("V")

plt.figure(3)
plt.subplot(221), plt.imshow(img_gray, cmap='gray'), plt.title("Gray")
plt.subplot(222), plt.imshow(Y, cmap='gray'), plt.title("Y")
plt.subplot(223), plt.imshow(Cr, cmap='gray'), plt.title("Cr")
plt.subplot(224), plt.imshow(Cb, cmap='gray'), plt.title("Cb")
#plt.show()

trsh = 130
trsh2 = 199

th, dst1 = cv2.threshold(G, trsh, 255, cv2.THRESH_BINARY_INV)  # Values under threshold become 255 (white)
th2, dst11 = cv2.threshold(G[0:2000, 0:1500], trsh2, 255, cv2.THRESH_BINARY)  # Values over threshold become 255(white)
dst2 = dst1.copy()
dst2[:2000, 0:1500] = dst11  # Selected region is replaced in original image

plt.figure(4)
plt.subplot(131), plt.imshow(dst1, cmap='gray'), plt.title("First threshold")
plt.subplot(132), plt.imshow(dst11, cmap='gray'), plt.title("Second threshold")
plt.subplot(133), plt.imshow(dst2, cmap='gray'), plt.title("Combined image")
#plt.show()

cv2.imwrite('segmented_8.jpg', dst2, [cv2.IMWRITE_JPEG_QUALITY, 80])  # Saving image

kernel = cv2.getStructuringElement(cv2.MORPH_DILATE, ksize=(4, 4))
dst2_dilate = cv2.dilate(dst2, kernel, iterations=7)  # Applying dilation operation

kernel1 = cv2.getStructuringElement(cv2.MORPH_ERODE, ksize=(4, 4))
dst2_erode = cv2.erode(dst2_dilate, kernel1, iterations=6)  # Applying erosion operation
cv2.imwrite('segmented_improved_8.jpg', dst2_erode, [cv2.IMWRITE_JPEG_QUALITY, 80])

plt.figure(5)
plt.subplot(131), plt.imshow(dst2, cmap='gray'), plt.title("Original Segmented Image")
plt.subplot(132), plt.imshow(dst2_dilate, cmap='gray'), plt.title("Dilated Segmented Image")
plt.subplot(133), plt.imshow(dst2_erode, cmap='gray'), plt.title("Eroded Segmented Image")
#plt.show()

# SECTION 2

no_labels, imLabels, statistics, centre = cv2.connectedComponentsWithStats(dst2_erode)  # Detecting components
print("Detected Objects", no_labels)

plt.figure(6)
plt.imshow(imLabels, cmap='gray'), plt.title('Labeled Image')
#plt.show()
cv2.imwrite('blobs_8.jpg', imLabels, [cv2.IMWRITE_JPEG_QUALITY, 80])

selected = np.zeros(imLabels.shape)  # Build  full black image
no_shapes = 0
for i in range(1, no_labels):  # Loop starts from 1, label 0 is the background
    height = statistics[i, cv2.CC_STAT_HEIGHT]  # Get the height statistics
    if 800 <= height < 900:  # Selecting only the heights that correspond to numbers
        selected[imLabels == i] = 255  # Selected component becomes white
        no_shapes = no_shapes + 1

plt.figure(7)
plt.imshow(selected, cmap='gray'), plt.title('Detected Numbers')
#plt.show()

img_selected = np.zeros_like(img)  # Convert the 1 channel image into 3 channels image
img_selected[:, :, 0] = selected
img_selected[:, :, 1] = selected
img_selected[:, :, 2] = selected

img_selected[np.where((img_selected == [255, 255, 255]).all(axis=2))] = [0, 0, 255]  # White pixels change color
img_selected = cv2.putText(img_selected, f'Counted Objects: {no_shapes}', (1950, 150), cv2.FONT_HERSHEY_COMPLEX,
                           3, color=(255, 255, 0), thickness=2)
plt.figure(8)
plt.imshow(img_selected), plt.imshow(img_selected)
#plt.show()
img_selected = cv2.cvtColor(img_selected, cv2.COLOR_BGR2RGB)
cv2.imwrite('valid_blobs_8.jpg', img_selected, [cv2.IMWRITE_JPEG_QUALITY, 80])

# SECTION 3

img1 = cv2.imread('books.jpg', 0)  # Reading image as grayscale
img2 = cv2.imread('casa.jpg', 0)

orbObj = cv2.ORB_create()  # Create ORB Object
kp1, descriptors1 = orbObj.detectAndCompute(img1, None)  # Computing descriptors and keypoints
kp2, descriptors2 = orbObj.detectAndCompute(img2, None)

brute_force = cv2.BFMatcher(cv2.NORM_HAMMING)  # ORB uses NORM_HAMMING
matched_descriptors = brute_force.match(descriptors1, descriptors2)  # Matching descriptors
matched_descriptors = sorted(matched_descriptors, key=lambda j: j.distance)  # Sorting the matches based on distance

matches = cv2.drawMatches(img1, kp1, img2, kp2, matched_descriptors[:20], None, flags=2)  # Drawing first 20 matches
# Flag =2 => shows only matched keypoints
cv2.imwrite('matched.jpg', matches, [cv2.IMWRITE_JPEG_QUALITY, 80])

plt.figure(9)
plt.imshow(matches), plt.title('Matched Objects')
plt.show()
