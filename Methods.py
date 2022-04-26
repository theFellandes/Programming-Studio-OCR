import numpy as np
from PIL import Image

class Moments:

    def __init__(self, instance):
        self.instance = instance

    def raw_moment(self, image, p, q):
        raw = 0
        for x in range(len(image)):
            for y in range(len(image)):
                raw += ((x + 1) ** p) * ((y + 1) ** q) * image[x][y]
        return raw

    def centroid_localization(self, image):
        x_prime = self.raw_moment(image, 1, 0) / self.raw_moment(image, 0, 0)
        y_prime = self.raw_moment(image, 0, 1) / self.raw_moment(image, 0, 0)
        return [x_prime, y_prime]

    def translation_invariance(self, image, p, q):
        trans_invar = 0
        center = self.centroid_localization(image)
        for x in range(len(image)):
            for y in range(len(image)):
                trans_invar += ((x - center[0]) ** p) * (y - center[1] ** q) * image[x][y]
        return trans_invar

    def scale_invariance(self, image, p, q):
        trans_invar = self.translation_invariance(image, p, q)
        weird_l = ((p + q) / 2) + 1
        return trans_invar / self.translation_invariance(image, 0, 0) ** weird_l

    def hu_moments(self, image):
        hu1 = self.scale_invariance(image, 2, 0) + self.scale_invariance(image, 0, 2)
        hu2 = (self.scale_invariance(image, 2, 0) - self.scale_invariance(image, 0, 2)) ** 2 + 4 * (
                self.scale_invariance(image, 1, 1) ** 2)
        hu3 = (self.scale_invariance(image, 3, 0) - 3 * self.scale_invariance(image, 1, 2)) ** 2 + (
                3 * self.scale_invariance(image, 2, 1) - self.scale_invariance(image, 0, 3)) ** 2
        hu4 = (self.scale_invariance(image, 3, 0) + self.scale_invariance(image, 1, 2)) ** 2 + (
                self.scale_invariance(image, 2, 1) + self.scale_invariance(image, 0, 3)) ** 2
        hu5 = (self.scale_invariance(image, 3, 0) - 3 * self.scale_invariance(image, 1, 2)) * (
                self.scale_invariance(image, 3, 0) + self.scale_invariance(image, 1, 2)) * (
                      (self.scale_invariance(image, 3, 0) + self.scale_invariance(image, 1, 2)) ** 2 - 3 * (
                      self.scale_invariance(image, 2, 1) + (self.scale_invariance(image, 0, 3) ** 2)))
        hu6 = (self.scale_invariance(image, 2, 0) - self.scale_invariance(image, 0, 2)) * (
                (self.scale_invariance(image, 3, 0) + self.scale_invariance(image, 1, 2)) ** 2 - (
                self.scale_invariance(image, 2, 1) + self.scale_invariance(image, 0,
                                                                           3) ** 2)) + 4 * self.scale_invariance(image,
                                                                                                                 1,
                                                                                                                 1) * (
                      (self.scale_invariance(image, 3, 0) + self.scale_invariance(image, 1, 2)) * (
                      self.scale_invariance(image, 2, 1) + self.scale_invariance(image, 0, 3)))
        hu7 = (3 * self.scale_invariance(image, 2, 1) - self.scale_invariance(image, 3, 0)) * (
                self.scale_invariance(image, 3, 0) + self.scale_invariance(image, 1, 2)) * (
                      (self.scale_invariance(image, 3, 0) + self.scale_invariance(image, 1, 2)) ** 2 - 3 * (
                      self.scale_invariance(image, 2, 1) + self.scale_invariance(image, 0, 3)) ** 2) + (
                      3 * self.scale_invariance(image, 2, 1) - self.scale_invariance(image, 0, 3)) * (
                      self.scale_invariance(image, 2, 1) + self.scale_invariance(image, 0, 3)) * (
                      (3 * (self.scale_invariance(image, 3, 0) + self.scale_invariance(image, 1, 2)) ** 2) - (
                      self.scale_invariance(image, 2, 1) + self.scale_invariance(image, 0, 3)) ** 2)
        hu = [hu1, hu2, hu3, hu4, hu5, hu6, hu7]
        for i in range(7):
            hu[i] = np.log10(abs(float(hu[i])))
        return hu

    def r_moments(self, hu):
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


class Label:

    def __init__(self, instance):
        self.instance = instance

    def np2PIL_color(self, im):
        # print("size of arr: ", im.shape)
        img = Image.fromarray(np.uint8(im))
        return img

    def threshold(self, im, T, LOW, HIGH):
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

    def update_array(self, a, label1, label2):
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

    def blob_coloring_8_connected(self, bim, ONE):
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
                            self.update_array(a, min_label, label_u)

                        if min_label != label_l and label_l != max_label:
                            self.update_array(a, min_label, label_l)

                        if min_label != label_ul and label_ul != max_label:
                            self.update_array(a, min_label, label_ul)

                        if min_label != label_ur and label_ur != max_label:
                            self.update_array(a, min_label, label_ur)

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





    def negative(self, image, musty):
        gray = image.convert('L')
        threshold = 200
        neg_bin = musty.binarize_image(gray, threshold, True)
        labeled_img, num_labels = musty.connected_component_labeling(neg_bin)
        img_colored_blobs = musty.blob_coloring(labeled_img, num_labels)
        img_colored_blobs.show()

        character = ""
        if num_labels - 1 == 0:
            character = 'C'
        if num_labels - 1 == 1:
            character = 'A'
        if num_labels - 1 == 2:
            character = 'B'
        return character

    def rectangle(self, label):
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

        if rectangles[0][2] == nrow - 1 and rectangles[0][3] == ncol - 1:
            rectangles = np.delete(rectangles, 0, 0)

        return rectangles
