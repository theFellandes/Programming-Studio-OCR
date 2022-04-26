from PIL import Image
import numpy as np

def main():
    # Read characters from an image file
    img = Image.open('./ABC-overlapped.jpg').convert('RGB')
    img_gray = img.convert('L')  # converts the image to grayscale image
    img_gray.show()  # show the image
    ONE = 200  # set value of 1-valued pixels
    a = np.asarray(img_gray)  # convert from PIL to np array
    a_bin = threshold(a, 150, ONE, 0)  # threshold the image a, with threshold T, LOW and HIGH
    im = Image.fromarray(a_bin)  # from np array to PIL format
    im.show()
    label = blob_coloring_8_connected(a_bin, ONE)  # labels 8-connected components
    new_img2 = np2PIL_color(label)  # converts from np array to PIL format
    new_img2.show()  # shows the image
    rects = rectangles(label)
    rects.show()


def binary_image(nrow, ncol, Value):
    # Creates artificial binary image array nrowxncol with values 0 and ONE
    # The image contains lines, rectangles and circles
    # used to test the labeling algorithms

    x, y = np.indices((nrow, ncol))
    mask_lines = np.zeros(shape=(nrow, ncol))

    x0, y0, r0 = 30, 30, 5
    x1, y1, r1 = 70, 30, 5

    for i in range(50, 70):
        mask_lines[i][i] = 1
        mask_lines[i][i + 1] = 1
        mask_lines[i][i + 2] = 1
        mask_lines[i][i + 3] = 1
        mask_lines[i][i + 6] = 1
        mask_lines[i - 20][90 - i + 1] = 1
        mask_lines[i - 20][90 - i + 2] = 1
        mask_lines[i - 20][90 - i + 3] = 1

    # mask_circle1 = np.abs((x - x0) ** 2 + (y - y0) ** 2 - r0 ** 2 ) <= 5
    mask_square1 = np.fmax(np.absolute(x - x1), np.absolute(y - y1)) <= r1
    # mask_square2 = np.fmax(np.absolute( x - x2), np.absolute( y - y2)) <= r2
    # mask_square3 = np.fmax(np.absolute( x - x3), np.absolute( y - y3)) <= r3
    # mask_square4 =  np.fmax(np.absolute( x - x4), np.absolute( y - y4)) <= r4
    # imge = np.logical_or ( np.logical_or(mask_lines, mask_circle1), mask_square1) * Value
    imge = np.logical_or(mask_lines, mask_square1) * Value
    # imge = np.logical_or(mask_lines, mask_circle1) * Value

    return imge


def np2PIL(im):
    print("size of arr: ", im.shape)
    img = Image.fromarray(im, 'RGB')
    return img


def np2PIL_color(im):
    print("size of arr: ", im.shape)
    img = Image.fromarray(np.uint8(im))
    return img


def threshold(im, T, LOW, HIGH):
    # threshold the image im, returns im_out
    # im_out = LOW, if im < T;   im_out = HIGH otherwise
    (nrows, ncols) = im.shape
    im_out = np.zeros(shape=im.shape)
    for i in range(nrows):
        for j in range(ncols):
            if abs(im[i][j]) < T:
                im_out[i][j] = LOW
            else:
                im_out[i][j] = HIGH
    return im_out


def blob_coloring_8_connected(bim, ONE):
    # labels binary image bim, where one-valued pixels have value ONE,
    # by using 8-connected component labeling algorithm.
    # Two-pass labeling is used
    max_label = int(10000)
    nrow = bim.shape[0]
    ncol = bim.shape[1]
    im = np.zeros(shape=(nrow, ncol), dtype=int)
    a = np.arange(0, max_label, dtype=int)
    color_map = np.zeros(shape=(max_label, 3), dtype=np.uint8)
    color_im = np.zeros(shape=(nrow, ncol, 3), dtype=np.uint8)

    for i in range(max_label):
        np.random.seed(i)
        color_map[i][0] = np.random.randint(0, 255, 1, dtype=np.uint8)
        color_map[i][1] = np.random.randint(0, 255, 1, dtype=np.uint8)
        color_map[i][2] = np.random.randint(0, 255, 1, dtype=np.uint8)

    k = 0
    for i in range(nrow):
        for j in range(ncol):
            im[i][j] = max_label
    for i in range(1, nrow - 1):
        for j in range(1, ncol - 1):
            c = bim[i][j]
            l = bim[i][j - 1]
            u = bim[i - 1][j]
            label_u = im[i - 1][j]
            label_l = im[i][j - 1]
            label_ul = im[i - 1][j - 1]
            label_ur = im[i - 1][j + 1]

            im[i][j] = max_label
            if c == ONE:
                min_label = min(label_u, label_l, label_ul, label_ur)
                if min_label == max_label:  # u = l = 0
                    k += 1
                    im[i][j] = k
                else:   # at least one of u or l or ul or ur is ONE
                    im[i][j] = min_label
                    if min_label != label_u and label_u != max_label:
                        update_array(a, min_label, label_u)

                    elif min_label != label_l and label_l != max_label:
                        update_array(a, min_label, label_l)

                    elif min_label != label_ul and label_ul != max_label:
                        update_array(a, min_label, label_ul)

                    elif min_label != label_ur and label_ur != max_label:
                        update_array(a, min_label, label_ur)

            else:
                im[i][j] = max_label
    for i in range(k + 1):
        index = i
        while a[index] != index:
            index = a[index]
        a[i] = a[index]

    for i in range(nrow):
        for j in range(ncol):

            if bim[i][j] == ONE:
                im[i][j] = a[im[i][j]]
                if im[i][j] == max_label:
                    im[i][j] == 0
                    color_im[i][j][0] = 0
                    color_im[i][j][1] = 0
                    color_im[i][j][2] = 0
                color_im[i][j][0] = color_map[im[i][j], 0]
                color_im[i][j][1] = color_map[im[i][j], 1]
                color_im[i][j][2] = color_map[im[i][j], 2]
    return color_im


def update_array(a, label1, label2):
    index = lab_small = lab_large = 0
    if label1 < label2:
        lab_small = label1
        lab_large = label2
    else:
        lab_small = label2
        lab_large = label1
    index = lab_large
    while index > 1 and a[index] != lab_small:
        if a[index] < lab_small:
            temp = index
            index = lab_small
            lab_small = a[temp]
        elif a[index] > lab_small:
            temp = a[index]
            a[index] = lab_small
            index = temp
        else:  # a[index] == lab_small
            break

    return


def rectangle_boundaries(label):
    nrow = label.shape[0]
    ncol = label.shape[1]



if __name__ == '__main__':
    main()
