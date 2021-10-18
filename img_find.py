import numpy as np
import img_rotation
import cv2

# def del_blank():
#     img = cv2.imread('sample_for_vision_3.jpg')
#     rotated_img = img_rotation.img_read(img_rotation.main())
#     imgray = cv2.cvtColor(rotated_img, cv2.COLOR_BGR2GRAY)
#     ret, thresh = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY)
#     thresh = 255 - thresh
#     contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#
#     cnt = contours[0]
#     x, y, w, h = cv2.boundingRect(cnt)
#     cv2.rectangle(rotated_img, (x, y), (x+w, y+h), (0, 0, 0), 3)
#
#     print(x, y, w, h)
#
#     cv2.imshow('ssss', rotated_img)
#     cv2.waitKey()
#
#     return y, y+h
#
#
# temp_img = img_rotation.img_read(img_rotation.main())
#
# start, end = del_blank()
# cut_img = temp_img[start:end]
# cv2.imshow('asdfasf', cut_img)
# cv2.waitKey()
# cv2.destroyAllWindows()
# np.save('cut.jpg', cut_img)
# cut_img = 'cut.jpg'
#
# img_rotation.draw_line(cut_img)
temp_img = 'sample_for_vision.png'
angle = img_rotation.find(temp_img)
temp_img = img_rotation.img_read(temp_img)
height, width, channel = temp_img.shape
print(angle)

length = 0
blank = 0
if height > width:
    length = height
    blank = int((height - width) / 2)
    temp_img = cv2.copyMakeBorder(temp_img, 0, 0, blank, blank, cv2.BORDER_CONSTANT, value=(255, 255, 255))
elif width > height:
    length = width
    blank = int((width - height) / 2)
    temp_img = cv2.copyMakeBorder(temp_img, blank, blank, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
print(length, width, height)
cv2.imshow('tkrkr', temp_img)
cv2.waitKey()

# original_rotated = img_rotation.img_rotate(temp_img, angle+180)
# original_rotated = img_rotation.img_read(original_rotated)

rot_mat = cv2.getRotationMatrix2D((length/2, length/2), angle+180, 1)
original_rotated = cv2.warpAffine(temp_img, rot_mat, (length, length), borderValue=(255, 255, 255))

cv2.imshow('final image', original_rotated)
cv2.waitKey()
cv2.destroyAllWindows()

# blank_img = np.ones((length, length), dtype=np.uint8) * 255
# cv2.imshow('white', blank_img)
# cv2.waitKey()
# cv2.destroyAllWindows()
