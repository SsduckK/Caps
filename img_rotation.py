import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def find(img_input):
    temp_img = img_input
    np_temp_img = cv2.imread(temp_img)
    # reverse_img = img_rotate(temp_img, 180)
    # rotation_img = img_rotate(temp_img, 45)
    # reverse_rot_img = img_rotate(temp_img, -45)
    # RA_img = img_rotate(temp_img, 90)
    # reverse_RA_img = img_rotate(temp_img, -90)
    # print(f'original:{variance(temp_img)}\nreverse:{variance(reverse_img)}\nrotated:{variance(rotation_img)}\nreverse:{variance(reverse_rot_img)}\n90:{variance(RA_img)}\n-90:{variance(reverse_RA_img)}')

    h_blank, w_blank = make_blank(temp_img)
    temp_img = cv2.copyMakeBorder(np_temp_img, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=(255, 0, 0))
    np.save('border.jpg', temp_img)
    temp_img = 'border.jpg'
    temp_img = img_resize(temp_img, h_blank, w_blank)
    np.save('rotated_img.jpg', temp_img)
    temp_img = 'rotated_img.jpg'
    reverse_img = img_rotate(temp_img, 0)
    # draw_line(temp_img)

    # print(rotation_img_data(temp_img, 8))
    angle, zero = first_max_line_img(reverse_img)
    print('first angle:', angle)
    find_img = img_rotate(reverse_img, angle)
    #draw_line(find_img)
    second_img = img_read(find_img)
    np.save('second_img.jpg', second_img)
    second_img = 'second_img.jpg'
    second_angle = second_max_line_img(second_img)
    print('second angle:', second_angle)
    final_img = img_rotate(second_img, second_angle)

    draw_line(final_img)
    return angle+second_angle


def img_read(img_input):
    img_original = cv2.imread(img_input)
    if img_input == 'rotated_img.jpg':
        img_original = np.load('rotated_img.jpg.npy')
    if img_input == 'border.jpg':
        img_original = np.load('border.jpg.npy')
    if img_input == 'second_img.jpg':
        img_original = np.load('second_img.jpg.npy')
    if img_input == 'final_img.jpg':
        img_original = np.load('final_img.jpg.npy')
    if img_input == 'cut.jpg':
        img_original = np.load('cut.jpg.npy')
    # img_resized = cv2.resize(img_original, dsize=(1440, 810))
    # return img_resized
    return img_original


def make_blank(img_input):
    img_original = img_read(img_input)
    height, width, channel = img_original.shape
    diameter = math.sqrt(height ** 2 + width ** 2)
    top_bottom_blank = (diameter - height) / 2
    side_blank = (diameter - width) / 2
    return top_bottom_blank, side_blank


def img_resize(img_input, top_bottom_blank=0, side_blank=0):
    img_original = img_read(img_input)
    #top_bottom_blank, side_blank = make_blank(img_input)
    img_extended = cv2.copyMakeBorder(img_original, int(top_bottom_blank), int(top_bottom_blank),
                                      int(side_blank), int(side_blank),
                                      cv2.BORDER_CONSTANT, value=(255, 255, 255))
    img_resized = cv2.resize(img_extended, dsize=(1080, 1080))
    return img_resized


def img_binary_img(img_input):
    img = img_resize(img_input)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_binary = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 75, 2)
    return img_binary


def pix_count(img_input):
    img = img_binary_img(img_input)
    height, width = img.shape
    y_pixel_list = np.zeros(height)
    for i in range(height):
        line = img[i:i + 1]
        non_black_pixel = np.count_nonzero(line)
        black_pixel = line.size - non_black_pixel
        y_pixel_list[i] = black_pixel
    x = np.arange(height)
    line_threshold = 100

    y_zero_list = []
    for i in range(height):
        if y_pixel_list[i] <= line_threshold:
            y_pixel_list[i] = 0
            y_zero_list.append(i)
    y_axis = y_pixel_list
    plt.bar(x, y_axis, color='blue')
    plt.hlines(line_threshold, x[0], x[-1], color='red')
    #plt.show()

    return y_zero_list


def draw_line(img_input):
    img = img_read(img_input)
    zero_list = pix_count(img_input)
    height, width, channel = img.shape
    for i in zero_list:
        img = cv2.line(img, (0, i), (width, i), (255, 0, 0), 1)
    cv2.imshow('img_final', img)
    cv2.waitKey()
    cv2.destroyAllWindows()


def variance(img_input):
    zero_list = pix_count(img_input)
    zero_numpy = np.array(zero_list)
    #return zero_numpy.var()
    if np.isnan(zero_numpy.var()):
        return 0
    else:
        return zero_numpy.var()


def img_rotate(img_input, angle=0):
    img = img_resize(img_input)
    height, width, channel = img.shape
    rotated_img = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
    done = cv2.warpAffine(img, rotated_img, (width, height), borderValue=(255, 255, 255))
    #cv2.imshow('img', done)
    cv2.waitKey()
    np.save('rotated_img.jpg', done)
    temp_img_name = 'rotated_img.jpg'
    return temp_img_name


def rotation_img_data(img_input, rotation_img_count=0):
    if rotation_img_count == 0:
        rotation_img_count = 1
    img_copy = img_input
    img_list = []
    img_dict = {'angle': [], 'zero_line': []}
    img_dict_numpy_angle = 0
    img_dict_numpy_zero_line = 0
    angle = int(360 / rotation_img_count)
    for i in range(rotation_img_count):
        rotated_img = img_rotate(img_copy, angle)
        #img_list.append(variance(rotated_img))
        #img_list.append(len(pix_count(rotated_img)))
        img_dict['angle'].append(angle*(i+1))
        img_dict['zero_line'].append(len(pix_count(rotated_img)))
        img_dict_numpy_angle = np.array(img_dict['angle'])
        img_dict_numpy_zero_line = np.array(img_dict['zero_line'])
    #return img_list
    return img_dict_numpy_angle, img_dict_numpy_zero_line


def first_max_line_img(img_input):
    angle, zero_line = rotation_img_data(img_input, 8)
    max_zero_line = zero_line.argmax()
    max_angle = angle[max_zero_line]
    # print(angle, '\n', zero_line)
    # print(max_angle, max_zero_line)
    return max_angle, max_zero_line


def second_max_line_img(img_input):
    # semi_angle_control = []
    # for i in range((angle-22), angle+23):
    #     semi_angle_control.append(i)
    # print(semi_angle_control)
    rotated_pix = []
    for i in range(-22, 23):
        img = img_rotate(img_input, i)
        pix = pix_count(img)
        rotated_pix.append(len(pix))
    num_rotated_pix = np.array(rotated_pix)
    return num_rotated_pix.argmax()-22

