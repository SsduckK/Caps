import numpy as np
import img_rotation
import img_transform
import cv2


def main():
    temp_img = 'Vimg2.jpg'
    img_transform.find_line(temp_img)
    temp_img = 'trans_img.jpg'
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

    rot_mat = cv2.getRotationMatrix2D((length/2, length/2), angle+180, 1)
    original_rotated = cv2.warpAffine(temp_img, rot_mat, (length, length), borderValue=(255, 255, 255))

    cv2.imshow('final image', original_rotated)
    cv2.waitKey()
    cv2.destroyAllWindows()
    cv2.imwrite('final_transed_img.jpg', original_rotated)


if __name__ == '__main__':
    main()