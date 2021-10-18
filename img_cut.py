#import pytesseract
import cv2
import numpy as np
#pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

# def main():
#     #print(pytesseract.image_to_string('sample_for_vision_2.jpg'))
#     large = cv2.imread('sample_for_vision_1.png')
#     rgb = cv2.pyrUp(large)
#     small = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
#     grad = cv2.morphologyEx(small, cv2.MORPH_GRADIENT, kernel)
#     _, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
#     connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
#     e_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))
#     d_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 3))
#     erode = cv2.erode(connected, e_kernel)
#     dilate = cv2.dilate(erode, d_kernel)
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
#     second_connected = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, kernel)
#     contours, hierarchy = cv2.findContours(second_connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#     mask = np.zeros(bw.shape, dtype=np.uint8)
#     cutting_img_list = []
#     cutting_img_count = 0
#     for idx in range(len(contours)):
#         x, y, w, h = cv2.boundingRect(contours[idx])
#         mask[y:y+h, x:x+w] = 0
#         cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)
#         r = float(cv2.countNonZero(mask[y:y+h, x:x+w])) / (w * h)
#         if r > 0.4 and w > 10 and h > 10:
#             cv2.rectangle(rgb, (x-2, y-10), (x+w, y+h+2), (0, 0, 255), 2)
#             print(idx, ':', x, y, w, h, '\n')
#             cut_img = rgb[y - 10:y + h, x - 10:x + w].copy()
#             cutting_img_list.append(cut_img)
#             cutting_img_count += 1
#
#     for i in range(1, cutting_img_count//10):
#         cv2.imshow(f"{i}", cutting_img_list[i])
#
#     # cv2.imshow('1', grad)
#     # cv2.imshow('2', bw)
#     # cv2.imshow('3', connected)
#     # cv2.imshow('4', erode)
#     # cv2.imshow('5', dilate)
#     # cv2.imshow('6', second_connected)
#     cv2.imshow('sample', rgb)
#     cv2.waitKey()


def main():
    large = cv2.imread('sample_for_vision_2.png')
    rgb = large
    small = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    grad = cv2.morphologyEx(small, cv2.MORPH_GRADIENT, kernel)
    _, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    # using RETR_EXTERNAL instead of RETR_CCOMP
    contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    mask = np.zeros(bw.shape, dtype=np.uint8)
    cutting_img_list = []
    cutting_img_count = 0
    for idx in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[idx])
        mask[y:y+h, x:x+w] = 0
        cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)
        r = float(cv2.countNonZero(mask[y:y+h, x:x+w])) / (w * h)
        if r > 0.45 and w > 8 and h > 8:
            #cv2.rectangle(rgb, (x, y), (x + w - 1, y + h - 1), (0, 255, 0), 2)
            print(idx, ':', x, y, w, h, '\n')
            cut_img = rgb[y:y + h, x-2:x + w + 2].copy()
            cutting_img_list.append(cut_img)
            cutting_img_count += 1

    # for i in range(1, cutting_img_count//5):
    #     cv2.imshow(f"{i}", cutting_img_list[i])
    # show image with contours rect
    # cv2.imshow('rects', rgb)

    erosion = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    sample = cutting_img_list[110]
    print(sample.shape)
    sample_gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
    cv2.imshow('agasdf', sample_gray)

    ret, dst = cv2.threshold(sample_gray, 0, 255, cv2.THRESH_OTSU)
    cv2.imshow('binary', dst)

    word_contours, word_hierachy = cv2.findContours(dst, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    for count in word_contours:
        x, y, w, h = cv2.boundingRect(count)
        cv2.rectangle(rgb, (x, y), (x + w, y + h), (255, 0, 0), 1)

    cv2.imshow('ori', rgb)
    cv2.imshow('contour', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()
    print(sample.shape)


if __name__ == "__main__":
    main()

