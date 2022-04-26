from PIL import Image, ImageDraw, ImageFont
import numpy as np
import tkinter.filedialog as filedialog
import tkinter as tk

def main():
    master = tk.Tk()
    top_frame = tk.Frame(master)
    bottom_frame = tk.Frame(master)
    line = tk.Frame(master, height=1, width=400, bg="grey80", relief='groove')
    input_path = tk.Label(top_frame, text="Input File Path:")
    input_path2 = tk.filedialog.askopenfilename()
    input_entry = tk.Entry(top_frame, text="", width=40)
    input_entry.delete(1, tk.END)  # Remove current text in entry
    input_entry.insert(0, input_path2)
    browse1 = tk.Button(top_frame, text="Browse")

    begin_button = tk.Button(bottom_frame, text='Begin!')

    top_frame.pack(side=tk.TOP)
    line.pack(pady=10)
    bottom_frame.pack(side=tk.BOTTOM)

    input_path.pack(pady=5)
    input_entry.pack(pady=5)
    browse1.pack(pady=5)

    begin_button.pack(pady=20, fill=tk.X)


    # Read characters from an image file
    img = Image.open(input_path2).convert('RGB')
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
    rects = rectangle(label)
    fontsize = 30
    font = ImageFont.truetype("arial.ttf", fontsize)
    imgdraw = ImageDraw
    for i in range(len(rects)):
        imgdraw.rectangle((rects[i][1], rects[i][0], rects[i][3], rects[i][2]), outline="red")
        # img_char = foo(label)
        imgdraw.text(((rects[i][1] + rects[i][3]) / 2, rects[i][0] - 25), "temporary", fill="red", font=font)
    img.show()


def binary_image(nrow, ncol, Value):
    # Creates artificial binary image array nrowxncol with values 0 and ONE
    # The image contains lines, rectangles and circles
    # used to test the labeling algorithms

    x, y = np.indices((nrow, ncol))
    mask_lines = np.zeros(shape=(nrow, ncol))

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
    mask_square1 = np.fmax(np.absolute(x - x1), np.absolute(y - y1)) <= r1
    imge = np.logical_or(mask_lines, mask_square1) * Value

    return imge


# printing binary image, coloured
def np2PIL_color(im):
    # print("size of arr: ", im.shape)
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
    max_label = int(10_000)
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
                else:  # at least one of u or l or ul or ur is ONE
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
                    color_im[i][j][0] = 0
                    color_im[i][j][1] = 0
                    color_im[i][j][2] = 0
                color_im[i][j][0] = color_map[im[i][j], 0]
                color_im[i][j][1] = color_map[im[i][j], 1]
                color_im[i][j][2] = color_map[im[i][j], 2]
    return color_im


def update_array(a, label1, label2):
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


# rectangle uses same principle as the 8-blob colouring
def rectangle(label):
    nrow = label.shape[0]
    ncol = label.shape[1]
    k = 0
    # Using dict, we can store (r, g, b) values in a key-value pair
    colors = dict()
    for i in range(nrow):
        for j in range(ncol):
            # same color belongs in the same k value
            rgb = (label[i][j][0], label[i][j][1], label[i][j][2])
            if rgb not in colors:
                colors[rgb] = k
                k += 1

    # filling rectangles with zeros and colors dictionary
    rectangles = np.zeros(shape=(len(colors), 4), dtype=int)
    for x in range(len(rectangles)):
        # temp min_y, min_x
        rectangles[x][0], rectangles[x][1] = 10_000, 10_000
    # y
    for i in range(nrow):
        # x
        for j in range(ncol):
            rgb = (label[i][j][0], label[i][j][1], label[i][j][2])
            # min_y
            if i < rectangles[colors[rgb]][0]:
                rectangles[colors[rgb]][0] = i
            # min_x
            if j < rectangles[colors[rgb]][1]:
                rectangles[colors[rgb]][1] = j
            # max_y
            if i > rectangles[colors[rgb]][2]:
                rectangles[colors[rgb]][2] = i
            # max_x
            if j > rectangles[colors[rgb]][3]:
                rectangles[colors[rgb]][3] = j

    # there's a boundary space outwards, removing that rectangle
    if rectangles[0][2] == nrow - 1 and rectangles[0][3] == ncol - 1:
        rectangles = np.delete(rectangles, 0, 0)
    return rectangles


def raw_moment(image, p, q):
    raw = 0
    for x in range(len(image)):
        for y in range(len(image)):
            raw += ((x + 1) ** p) * ((y + 1) ** q) * image[x][y]
    return raw


def centroid_localization(image):
    x_prime = raw_moment(image, 1, 0) / raw_moment(image, 0, 0)
    y_prime = raw_moment(image, 0, 1) / raw_moment(image, 0, 0)
    return [x_prime, y_prime]


def translation_invariance(image, p, q):
    trans_invar = 0
    center = centroid_localization(image)
    for x in range(len(image)):
        for y in range(len(image)):
            trans_invar += ((x - center[0]) ** p) * (y - center[1] ** q) * image[x][y]
    return trans_invar


def scale_invariance(image, p, q):
    trans_invar = translation_invariance(image, p, q)
    weird_l = ((p + q) / 2) + 1
    return trans_invar / translation_invariance(image, 0, 0) ** weird_l


def hu_moments(image):
    hu1 = scale_invariance(image, 2, 0) + scale_invariance(image, 0, 2)
    hu2 = (scale_invariance(image, 2, 0) - scale_invariance(image, 0, 2)) ** 2 + 4 * (
                scale_invariance(image, 1, 1) ** 2)
    hu3 = (scale_invariance(image, 3, 0) - 3 * scale_invariance(image, 1, 2)) ** 2 + (
                3 * scale_invariance(image, 2, 1) - scale_invariance(image, 0, 3)) ** 2
    hu4 = (scale_invariance(image, 3, 0) + scale_invariance(image, 1, 2)) ** 2 + (
                scale_invariance(image, 2, 1) + scale_invariance(image, 0, 3)) ** 2
    hu5 = (scale_invariance(image, 3, 0) - 3 * scale_invariance(image, 1, 2)) * (
                scale_invariance(image, 3, 0) + scale_invariance(image, 1, 2)) * (
                      (scale_invariance(image, 3, 0) + scale_invariance(image, 1, 2)) ** 2 - 3 * (
                          scale_invariance(image, 2, 1) + (scale_invariance(image, 0, 3) ** 2)))
    hu6 = (scale_invariance(image, 2, 0) - scale_invariance(image, 0, 2)) * (
                (scale_invariance(image, 3, 0) + scale_invariance(image, 1, 2)) ** 2 - (
                    scale_invariance(image, 2, 1) + scale_invariance(image, 0, 3) ** 2)) + 4 * scale_invariance(image,
                                                                                                                1,
                                                                                                                1) * (
                      (scale_invariance(image, 3, 0) + scale_invariance(image, 1, 2)) * (
                          scale_invariance(image, 2, 1) + scale_invariance(image, 0, 3)))
    hu7 = (3 * scale_invariance(image, 2, 1) - scale_invariance(image, 3, 0)) * (
                scale_invariance(image, 3, 0) + scale_invariance(image, 1, 2)) * (
                      (scale_invariance(image, 3, 0) + scale_invariance(image, 1, 2)) ** 2 - 3 * (
                          scale_invariance(image, 2, 1) + scale_invariance(image, 0, 3)) ** 2) + (
                      3 * scale_invariance(image, 2, 1) - scale_invariance(image, 0, 3)) * (
                      scale_invariance(image, 2, 1) + scale_invariance(image, 0, 3)) * (
                      (3 * (scale_invariance(image, 3, 0) + scale_invariance(image, 1, 2)) ** 2) - (
                          scale_invariance(image, 2, 1) + scale_invariance(image, 0, 3)) ** 2)
    return [hu1, hu2, hu3, hu4, hu5, hu6, hu7]


def r_moments(hu):
    r1 = np.sqrt(hu[1]) / hu[0]
    r2 = (hu[0] + np.sqrt(hu[1])) / (hu[0] - np.sqrt(hu[1]))
    r3 = np.sqrt(hu[2]) / np.sqrt(hu[3])
    r4 = np.sqrt(hu[2]) / np.sqrt(np.abs(hu[4]))
    r5 = np.sqrt(hu[3]) / np.sqrt(np.abs(hu[4]))
    r6 = np.abs(hu[5]) / (hu[0] * hu[2])
    r7 = np.abs(hu[5]) / (hu[0] * np.sqrt(np.abs(hu[4])))
    r8 = np.abs(hu[5]) / (hu[2] * np.sqrt(hu[1]))
    r9 = np.abs(hu[5]) / np.sqrt((hu[1] * np.abs(hu[4])))
    r10 = np.abs(hu[4]) / (hu[2] * hu[3])
    return [r1, r2, r3, r4, r5, r6, r7, r8, r9, r10]


def foo(image):
    func = r_moments(hu_moments(image))
    return str(index_to_char(func))


def index_to_char(index):
    if index >= 10:
        char = chr(int(index + 55))
    else:
        char = index
    return str(char)


if __name__ == '__main__':
    main()
