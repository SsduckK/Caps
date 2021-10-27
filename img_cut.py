import pytesseract
import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from googletrans import Translator
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
    large = cv2.imread('final_transed_img.jpg')
    rgb = large
    sharpening = np.array([[-1, -1, -1, -1, -1],
                         [-1, 2, 2, 2, -1],
                         [-1, 2, 9, 2, -1],
                         [-1, 2, 2, 2, -1],
                         [-1, -1, -1, -1, -1]]) / 9.0
    rgb_copy = rgb.copy()
    rgb_copy = cv2.filter2D(rgb_copy, -1, sharpening)
    small = cv2.cvtColor(rgb_copy, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    grad = cv2.morphologyEx(small, cv2.MORPH_GRADIENT, kernel)
    _, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # cv2.imshow('bw', bw)
    # cv2.waitKey()
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 2))
    connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    # using RETR_EXTERNAL instead of RETR_CCOMP
    contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    mask = np.zeros(bw.shape, dtype=np.uint8)
    cutting_img_list = []
    point_list = []
    cutting_img_count = 0

    trans = Translator()

    img_pil = Image.fromarray(rgb_copy)
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype("fonts/NanumGothic.ttf", 10)

    for idx in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[idx])
        mask[y:y+h, x:x+w] = 0
        cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)
        r = float(cv2.countNonZero(mask[y:y+h, x:x+w])) / (w * h)
        if r > 0.45 and w > 8 and h > 8:
            point = [x, y]
            point_list.append(point)
            # cv2.rectangle(rgb_copy, (x, y), (x + w - 1, y + h - 1), (0, 255, 0), 1)
            # print(idx, ':', x, y, w, h, '\n')
            # cv2.imshow('rgb_copy', rgb_copy)
            # cv2.waitKey()
            draw.rectangle(((x, y), (x + w - 1, y + h - 1)), outline=(0, 255, 0), width=1)
            cut_img = rgb[y:y + h, x-2:x + w + 2].copy()
            cut_img = cv2.copyMakeBorder(cut_img, 8, 8, 12, 12, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            gray_img = cv2.cvtColor(cut_img, cv2.COLOR_BGR2GRAY)
            ret, dst = cv2.threshold(gray_img, 100, 255, cv2.THRESH_TOZERO)
            # cv2.imshow('asdf', dst)
            # cv2.waitKey()
            text = pytesseract.image_to_string(dst)
            text = text[:-2]
            # print(text)
            # transed = trans.translate(text, dest='ko')
            # print(transed.text)
            try:
                transed = trans.translate(text, dest='ko')
                # print(transed.text)
                text = text + '=' + transed.text
                print(text)
                draw.text((x, y - 10), text, font=font, fill=(255, 0, 0))
            except:
                print('error')
            # cv2.imshow('asddf', rgb_copy)
            # cv2.waitKey()
            # cutting_img_list.append(cut_img)
            # cutting_img_count += 1
    # text = []
    # for i in range(1, cutting_img_count):
    #     cutting_img_list[i] = cv2.copyMakeBorder(cutting_img_list[i], 8, 8, 8, 8, cv2.BORDER_CONSTANT,
    #                                              value=(255, 255, 255))
    #     gray_img = [0 for i in range(len(cutting_img_list))]
    #     dst = [0 for i in range(len(cutting_img_list))]
    #     # cv2.imshow(f"{i}", cutting_img_list[i])
    #     gray_img[i] = cv2.cvtColor(cutting_img_list[i], cv2.COLOR_BGR2GRAY)
    #     # cv2.imshow('cutgray', gray_img[i])
    #     cut_height, cut_width = gray_img[i].shape
    #     # print(cut_height, cut_width)
    #     box = int(cut_width/10)
    #     if (box % 2) == 0:
    #         box = box + 1
    #     elif box == 1:
    #         box = 3
    #     # print('box', box)
    #     ret, dst[i] = cv2.threshold(gray_img[i], 100, 255, cv2.THRESH_TOZERO)
    #     # dst[i] = cv2.adaptiveThreshold(gray_img[i], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, box, 10)
    #     # cv2.imshow('dst', dst[i])
    #     text.append(pytesseract.image_to_string(dst[i]))
    #     # print(text)
    #     # cv2.waitKey()
    #
    # # cv2.putText(rgb, text, (100, 100), cv2.FONT_ITALIC, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    #
    # trans = Translator()
    #
    # img_pil = Image.fromarray(rgb_copy)
    # draw = ImageDraw.Draw(img_pil)
    # font = ImageFont.truetype("fonts/NanumGothic.ttf", 10)
    #
    # for box_text, box_point_list in zip(text, point_list):
    #     box_text = box_text[:-2]
    #     transed = 0
    #     try:
    #         transed = trans.translate(box_text, dest='ko')
    #         # print(transed.text)
    #         box_text = box_text + '=' + transed.text
    #     except TypeError as te:
    #         print('unknown')
    #     draw.text((box_point_list[0], box_point_list[1]-10), box_text, font=font, fill=(255, 0, 0))
    #     #box_text = box_text + ' = ' + transed
    #     # cv2.putText(rgb_copy, box_text, (box_point_list[0], box_point_list[1]), cv2.FONT_ITALIC, 0.5, (255, 0, 0))

    rgb_copy = np.array(img_pil)
    # print(text)
    # print(point_list)
    # show image with contours rect
    # cv2.imshow('rects', rgb)
    #
    # erosion = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # sample = cutting_img_list[110]
    # print(sample.shape)
    # sample_gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('agasdf', sample_gray)
    #
    # ret, dst = cv2.threshold(sample_gray, 0, 255, cv2.THRESH_OTSU)
    # cv2.imshow('binary', dst)
    #
    # word_contours, word_hierachy = cv2.findContours(dst, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    #
    # for count in word_contours:
    #     x, y, w, h = cv2.boundingRect(count)
    #     cv2.rectangle(rgb, (x, y), (x + w, y + h), (255, 0, 0), 1)
    #


    cv2.imshow('ori', rgb_copy)
    cv2.imwrite('boximg.jpg', rgb_copy)
    # cv2.imshow('contour', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()
    # print(sample.shape)


if __name__ == "__main__":
    main()

