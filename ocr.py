from PIL import Image, ImageDraw, ImageFont
import numpy as np
from Classes import Musty, Label


class Ocr:

    def __init__(self, path):
        self.path = path

    def main(self):

        instance = Label(self)
        musty = Musty()
        # Read characters from an image file
        img = Image.open(self.path).convert('RGB')
        output_file = open("Output.txt", "w+")
        img_gray = img.convert('L')  # converts the image to grayscale image
        img_gray.show()  # show the image
        threshold = 100
        light_bg = True
        img_bin = musty.binarize_image(img_gray, threshold, light_bg)
        img_bin.show()
        fontsize = 30
        font = ImageFont.truetype("arial.ttf", fontsize)
        labeled_img, num_labels = musty.connected_component_labeling(img_bin)
        # assign a random color to each connected component
        img_colored_blobs = musty.blob_coloring(labeled_img, num_labels)
        img_colored_blobs.show()
        output_file.write('Number of characters in the image: ' + str(num_labels) + '\n')
        rects = instance.rectangle(np.asarray(img_colored_blobs))
        imgdraw = ImageDraw.Draw(img)
        letter_freq = dict()
        for i in range(len(rects)):
            imgdraw.rectangle((rects[i][1], rects[i][0], rects[i][3], rects[i][2]), outline="red")
            cropped_image = img_colored_blobs.crop((rects[i][1] - 2, rects[i][0] - 2, rects[i][3] + 2, rects[i][2] + 2))

            cropped_img_copy = cropped_image.copy().convert('L')
            cropped_arr = np.asarray(cropped_img_copy).copy()
            cropped_arr_img = instance.overlap(cropped_arr, musty=musty)

            recognized_character = instance.negative(cropped_arr_img, musty=musty)
            imgdraw.text(((rects[i][1] + rects[i][3]) / 2, rects[i][0] - 22), recognized_character, fill="red",
                         font=font)
            if recognized_character in letter_freq:
                letter_freq[recognized_character] += 1
            else:
                letter_freq[recognized_character] = 1
            output_file.write(recognized_character + '\n')
        output_file.write(str(letter_freq))
        img.show()

    if __name__ == '__main__':
        main()
