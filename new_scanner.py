# importing the necessary libraries

import numpy as np
import cv2
import imutils

# reading the image
image = cv2.imread("images/page.jpg")

# resizing the image
ratio = image[0] / 500.0
orig = image.copy()
image = imutils.resize(image, height=500)

# converting image into grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
cv2.imshow("gray", gray)
cv2.waitKey(0)

# using canny edge detector to identify the edges
edged = cv2.Canny(gray, 0, 50)
cv2.imshow("canny", edged)
cv2.waitKey(0)

# using findcontours to find the edges and corners
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# taking the maximum 4 values
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02*peri, True)

    if len(approx) == 4:
        screenCnt = approx
        break

cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
cv2.imshow("contours", image)
cv2.waitKey(0)
cv2.imshow("Outline", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# setting the contours in order of top left(tl), top right(tr), bottom left(bl), bottom right(br) as required in the perspective transform
tl = screenCnt[1][0]
tr = screenCnt[0][0]
bl = screenCnt[2][0]
br = screenCnt[3][0]

cv2.drawContours(image, screenCnt, -1, (0, 255, 0), 3)
gray = np.float32(gray)


# applying the perspective transform to get the birds eye view
pts1 = np.float32([[tl[0], tl[1]], [tr[0], tr[1]],
                   [bl[0], bl[1]], [br[0], br[1]]])
pts2 = np.float32([[0, 0], [600, 0], [0, 600], [600, 600]])
matrix = cv2.getPerspectiveTransform(pts1, pts2)
result = cv2.warpPerspective(image, matrix, (600, 600))
# printing the final result
cv2.imshow("final result", result)
cv2.waitKey(0)
