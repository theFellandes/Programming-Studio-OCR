import numpy as np
from PIL import Image, ImageOps


class Label:

    def __init__(self, instance):
        self.instance = instance

    def overlap(self, label, musty):
        nrow = label.shape[0]
        ncol = label.shape[1]
        colors = list()
        for i in range(nrow):
            for j in range(ncol):
                if label[i][j] != 0 and label[i][j] not in colors:
                    colors.append(label[i][j])
        for i in range(nrow):
            for j in range(ncol):
                if label[i][j] != colors[0]:
                    label[i][j] = 0
        img = Image.fromarray(label).convert('L')
        neg_img = ImageOps.invert(img)
        neg_bin = musty.binarize_image(neg_img, 50, True)
        return neg_bin

    def negative(self, image, musty):
        img_arr = ImageOps.invert(image.convert('L'))
        labeled_img, num_labels = musty.connected_component_labeling(img_arr)
        # img_colored_blobs = musty.blob_coloring(labeled_img, num_labels)
        # img_colored_blobs.show()

        width, height = img_arr.size
        # print(img_arr.getpixel((width / 2, height / 2)))
        character = ""
        if num_labels - 1 == 0:
            if img_arr.getpixel((width / 2, height / 2)) != 0:
                character = 'C'
            else:
                character = '1'

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


class Musty:
    # BINARIZATION
    # -----------------------------------------------------------------------------------
    # Function for creating and returning a binary image by thresholding a given
    # grayscale image
    def binarize_image(self, img_gray, threshold, light_bg):
        # create a binary image with default color = 0 (black) for each pixel
        img_bin = Image.new('1', img_gray.size)
        width, height = img_gray.size
        # for each pixel in the given grayscale image
        for y in range(height):
            for x in range(width):
                # get the value of the pixel
                pixel = img_gray.getpixel((x, y))
                # invert the image if it has dark characters on a light background
                max_pixel_value = 255
                if light_bg:
                    pixel = max_pixel_value - pixel
                # if the pixel has a greater value than the given threshold
                # set the corresponding pixel in binary image as 1 (color white)
                if pixel > threshold:
                    img_bin.putpixel((x, y), 1)
        # return the resulting binary image
        return img_bin

    # CONNECTED COMPONENT LABELING AND BLOB COLORING
    # -----------------------------------------------------------------------------------
    # Function for labeling connected components (characters) in a binary image
    def connected_component_labeling(self, img_bin):
        width, height = img_bin.size
        # initially all the pixels in the image are labeled as 0 (background)
        labels = np.zeros([height, width], dtype=int)
        # min_equivalent_labels list is used to store min equivalent label for each label
        min_equivalent_labels = []
        # labeling starts with 1 (as 0 is considered as the background of the image)
        current_label = 1
        # first pass to assign initial labels and determine minimum equivalent labels
        # from conflicts for each pixel in the given binary image
        # --------------------------------------------------------------------------------
        for y in range(height):
            for x in range(width):
                # get the value of the pixel
                pixel = img_bin.getpixel((x, y))
                # skip pixels with value 0 (background pixels)
                if pixel == 0:
                    continue
                # get the set of neighboring labels for this pixel
                # using get_neighbor_labels function defined below
                neighbor_labels = self.get_neighbor_labels(labels, (x, y))
                # if all the neighbor pixels are background pixels
                if len(neighbor_labels) == 0:
                    # assign current_label as the label of this pixel
                    # and increase current_label by 1
                    labels[y, x] = current_label
                    current_label += 1
                    # initially the minimum equivalent label is the label itself
                    min_equivalent_labels.append(labels[y, x])
                # if there is at least one non-background neighbor
                else:
                    # assign minimum neighbor label as the label of this pixel
                    labels[y, x] = min(neighbor_labels)
                    # a conflict occurs if there are multiple (different) neighbor labels
                    if len(neighbor_labels) > 1:
                        labels_to_merge = set()
                        # add min equivalent label for each neighbor to labels_to_merge set
                        for l in neighbor_labels:
                            labels_to_merge.add(min_equivalent_labels[l - 1])
                        # update minimum equivalent labels related to this conflict
                        # using update_equivalent_labels function defined below
                        self.update_min_equivalent_labels(min_equivalent_labels, labels_to_merge)
        # second pass to rearrange equivalent labels so they all have consecutive values
        # starting from 1 and assign min equivalent label of each pixel as its own label
        # --------------------------------------------------------------------------------
        # rearrange min equivalent labels using rearrange_min_equivalent_labels function
        self.rearrange_min_equivalent_labels(min_equivalent_labels)
        # for each pixel in the given binary image
        for y in range(height):
            for x in range(width):
                # get the value of the pixel
                pixel = img_bin.getpixel((x, y))
                # skip pixels with value 0 (background pixels)
                if pixel == 0:
                    continue
                # assign minimum equivalent label of each pixel as its own label
                labels[y, x] = min_equivalent_labels[labels[y, x] - 1]
        # return the labels matrix and the number of different labels
        return labels, len(set(min_equivalent_labels))

    # Function for getting labels of the neighbors of a given pixel
    def get_neighbor_labels(self, label_values, pixel_indices):
        x, y = pixel_indices
        # using a set to store different neighbor labels without any duplicates
        neighbor_labels = set()
        # add upper pixel to the set if the current pixel is not in the first row of the
        # image and its value is not zero (not a background pixel)
        if y != 0:
            u = label_values[y - 1, x]
            if u != 0:
                neighbor_labels.add(u)
        # add left pixel to the set if the current pixel is not in the first column of
        # the image and its value is not zero (not a background pixel)
        if x != 0:
            l = label_values[y, x - 1]
            if l != 0:
                neighbor_labels.add(l)
        # return the set of neighbor labels
        if y != 0 and x != 0:
            ul = label_values[y - 1, x - 1]
            if ul != 0:
                neighbor_labels.add(ul)

        if y != 0 and x != len(label_values[0]) - 1:
            ur = label_values[y - 1, x + 1]
            if ur != 0:
                neighbor_labels.add(ur)

        return neighbor_labels

    # Function for updating min equivalent labels by merging conflicting neighbor labels
    # as the smallest value among their min equivalent labels
    def update_min_equivalent_labels(self, all_min_eq_labels, min_eq_labels_to_merge):
        # find the min value among conflicting neighbor labels
        min_value = min(min_eq_labels_to_merge)
        # for each minimum equivalent label
        for index in range(len(all_min_eq_labels)):
            # if its value is in min_eq_labels_to_merge
            if all_min_eq_labels[index] in min_eq_labels_to_merge:
                # update its value as the min_value
                all_min_eq_labels[index] = min_value

    # Function for rearranging min equivalent labels so they all have consecutive values
    # starting from 1
    def rearrange_min_equivalent_labels(self, min_equivalent_labels):
        # find different values of min equivalent labels and sort them in increasing order
        different_labels = set(min_equivalent_labels)
        different_labels_sorted = sorted(different_labels)
        # create an array for storing new (consecutive) values for min equivalent labels
        new_labels = np.zeros(max(min_equivalent_labels) + 1, dtype=int)
        count = 1  # first label value to assign
        # for each different label value (sorted in increasing order)
        for l in different_labels_sorted:
            # determine the new label
            new_labels[l] = count
            count += 1  # increase count by 1 so that new label values are consecutive
        # assign new values of each minimum equivalent label
        for ind in range(len(min_equivalent_labels)):
            old_label = min_equivalent_labels[ind]
            new_label = new_labels[old_label]
            min_equivalent_labels[ind] = new_label

    # Function for blob (connected component) coloring (assigning random colors to
    # connected components)
    def blob_coloring(self, labels, num_labels):
        rgb_values = []
        max_color_value = 255
        # determine a random color for each different label
        for l in range(num_labels):
            while True:
                current_rgb = np.random.rand(3) * max_color_value
                # for ensuring each blob is light enough to distinguish from the background
                if sum(current_rgb) / 3 > 100:
                    break
            rgb_values.append(current_rgb)
        height, width = labels.shape
        # create a 3-d array for the color image with randomly colored blobs
        colored_blobs = np.zeros([height, width, 3], dtype=np.uint8)
        # assign randomly generated color of each different label value in labels matrix
        for y in range(height):
            for x in range(width):
                label_value = labels[y, x]
                # skip background
                if label_value == 0:
                    continue
                colored_blobs[y, x] = rgb_values[label_value - 1]
                # convert numpy array to a PIL image and return the image with colored blobs
        img_colored_blobs = Image.fromarray(colored_blobs.astype('uint8'))
        return img_colored_blobs


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
