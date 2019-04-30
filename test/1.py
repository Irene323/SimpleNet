import cv2

num = 141855
imageID0 = int(int(num) / 256)
patchID0 = int(num) % 256
image_dir0 = 'D:\\dataset\\liberty\\' + ('patches%04d.bmp' % (imageID0))
img = cv2.imread(image_dir0)
img = img[(int(patchID0 / 16) * 64):(int(patchID0 / 16) * 64 + 64), ((patchID0 % 16) * 64):((patchID0 % 16) * 64 + 64)]
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
# sift = cv2.SIFT()
# kp = sift.detect(gray, None)
# print(kp)
#
# img = cv2.drawKeypoints(gray, kp)
#
# cv2.imwrite('sift_keypoints.jpg', img)

