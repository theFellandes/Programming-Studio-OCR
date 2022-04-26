# IMPORTED MODULES
#-----------------------------------------------------------------------------------
from PIL import Image  # Python Imaging Library (PIL) modules
import numpy as np   # fundamental Python module for scientific computing
import os   # os module can be used for file and directory operations

# MAIN FUNCTION OF THE PROGRAM
#-----------------------------------------------------------------------------------
# Main function where this python script starts execution
class Test:
    def start(self):
        # path of the current directory where this program file is placed
        curr_dir = os.path.dirname(os.path.realpath(__file__))
        # read color image as a grayscale image
        img_file = curr_dir + '/ABC-overlapped.jpg'
        img_gray = Image.open(img_file).convert('L')
        img_gray.show()  # display the grayscale image
        # binarize the image based on thresholding and whether the background is lighter
        # or darker than the characters
        threshold = 100
        light_bg = True
        img_bin = self.binarize_image(img_gray, threshold, light_bg)
        img_bin.show()  # display the binary image
        # find and label 4-connected components in the binary image
        labeled_img, num_labels = self.connected_component_labeling(img_bin)
        print('Number of characters in the image: ' + str(num_labels))
        # assign a random color to each connected component
        img_colored_blobs = self.blob_coloring(labeled_img, num_labels)
        img_colored_blobs.show()  # display the randomly colored components

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

    # start() function is specified as the entry point(main function) where the program
    # starts execution
    if __name__ == '__main__':
        start()