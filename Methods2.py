import numpy as np
from PIL import ImageOps


class Label:

    def __init__(self, instance):
        self.instance = instance

    def negative(self, image, musty):
        img_arr = ImageOps.invert(image.convert('L'))
        labeled_img, num_labels = musty.connected_component_labeling(img_arr)
        # img_colored_blobs = musty.blob_coloring(labeled_img, num_labels)
        # img_colored_blobs.show()

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
