import cv2
import numpy as np
import itertools

img = cv2.imread('train_houghtransform3.jpg')
img = cv2.pyrDown(img)
img = cv2.pyrDown(img)
img_copy = img.copy()
height, width, channel = img_copy.shape

cv2.imshow('ori', img_copy)

print(width, height, channel)
img_gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
img_gray_copy = cv2.GaussianBlur(img_gray, (3, 3), 0)
cv2.imshow('gray', img_gray_copy)
edges = cv2.Canny(img_gray, 2500, 2700, apertureSize=5)
cv2.imshow('edge', edges)
cv2.waitKey()

lines = cv2.HoughLines(edges, 1, 0.005, 200)
list_point = []
for i in range(len(lines)):
    for rho, theta in lines[i]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        temp_list = [x1, x2, y1, y2]
        list_point.append(temp_list)
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
cv2.imshow('line', img)
cv2.waitKey()


def find_crosspoint(listA, listB):
    listA_copy, listB_copy = listA, listB
    x11, x12, y11, y12 = listA_copy[0], listA_copy[1], listA_copy[2], listA_copy[3]
    x21, x22, y21, y22 = listB_copy[0], listB_copy[1], listB_copy[2], listB_copy[3]
    m1 = (y12 - y11) / (x12 - x11)
    m2 = (y22 - y21) / (x22 - x21)
    m1 = round(m1, 3)
    m2 = round(m2, 3)
    x = (x11 * m1 - y11 - x21 * m2 + y21) / (m1 - m2)
    y = m1 * (x - x11) + y11
    return x, y

cross_point = list(itertools.combinations(list_point, 2))
points = []
for coord in cross_point:
    x, y = coord
    x, y = find_crosspoint(x, y)
    list_cross_point = []
    list_cross_point.append(x)
    list_cross_point.append(y)
    points.append(list_cross_point)

# x, y = find_crosspoint(list_point[0], list_point[1])
# list_cross_point.append(x)
# list_cross_point.append(y)
# cross_point.append(list_cross_point)
# list_cross_point =[]
# x, y = find_crosspoint(list_point[0], list_point[2])
# list_cross_point.append(x)
# list_cross_point.append(y)
# cross_point.append(list_cross_point)
# list_cross_point =[]
# x, y = find_crosspoint(list_point[0], list_point[3])
# list_cross_point.append(x)
# list_cross_point.append(y)
# cross_point.append(list_cross_point)
# list_cross_point =[]
# x, y = find_crosspoint(list_point[1], list_point[2])
# list_cross_point.append(x)
# list_cross_point.append(y)
# cross_point.append(list_cross_point)
# list_cross_point =[]
# x, y = find_crosspoint(list_point[1], list_point[3])
# list_cross_point.append(x)
# list_cross_point.append(y)
# cross_point.append(list_cross_point)
# list_cross_point =[]
# x, y = find_crosspoint(list_point[2], list_point[3])
# list_cross_point.append(x)
# list_cross_point.append(y)
# cross_point.append(list_cross_point)
# list_cross_point =[]

img_cross_point = []
for cross in points:
    if (0 < cross[0] < width) and (0 < cross[1] < height):
        img_cross_point.append(cross)
for cross in img_cross_point:
    x, y = cross
    cv2.circle(img, (int(x), int(y)), 2, (255, 0, 0), 2)
cv2.imshow('circle', img)

direction_list = []
for points in img_cross_point:
    x, y = points
    xy_sum = []
    xy_sum.append(x + y)
    xy_sum.append(x - y)
    xy_sum.append(x)
    xy_sum.append(y)
    direction_list.append(xy_sum)
print(direction_list)
right_bottom_x, right_bottom_y = max(direction_list)[2], max(direction_list)[3]
left_top_x, left_top_y = min(direction_list)[2], min(direction_list)[3]
edit_list = []
for direct in direction_list:
    edit_list.append(direct[1:])

left_bottom_x, left_bottom_y = min(edit_list)[1], min(edit_list)[2]
right_top_x, right_top_y = max(edit_list)[1], max(edit_list)[2]

pts1 = np.float32([[left_top_x, left_top_y], [left_bottom_x, left_bottom_y],
                  [right_bottom_x, right_bottom_y], [right_top_x, right_top_y]])
print(pts1)
pts2 = np.float32([[0, 0], [0, height - 1], [width - 1, height - 1], [width - 1, 0]])

mat = cv2.getPerspectiveTransform(pts1, pts2)
dst = cv2.warpPerspective(img, mat, (width, height))

cv2.imshow('trans', dst)
cv2.waitKey()
cv2.destroyAllWindows()
