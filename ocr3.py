from PIL import Image, ImageDraw
import numpy as np
from Methods2 import Label
from test import Test

class Ocr:

    def __init__(self, path):
        self.path = path

    def main(self):

        instance = Label(self)
        musty = Test()
        # Read characters from an image file
        img = Image.open(self.path).convert('RGB')
        img_gray = img.convert('L')  # converts the image to grayscale image
        img_gray.show()  # show the image
        threshold = 100
        light_bg = True
        img_bin = musty.binarize_image(img_gray, threshold, light_bg)
        img_bin.show()
        labeled_img, num_labels = musty.connected_component_labeling(img_bin)
        # assign a random color to each connected component
        img_colored_blobs = musty.blob_coloring(labeled_img, num_labels)
        img_colored_blobs.show()
        rects = instance.rectangle(np.asarray(img_colored_blobs))
        imgdraw = ImageDraw.Draw(img)
        for i in range(len(rects)):
            imgdraw.rectangle((rects[i][1], rects[i][0], rects[i][3], rects[i][2]), outline="red")
            cropped_image = img_bin.crop((rects[i][1] - 2, rects[i][0] - 2, rects[i][3] + 2, rects[i][2] + 2))
            recognized_character = instance.negative(cropped_image, musty=musty)
            imgdraw.text(((rects[i][1] + rects[i][3]) / 2, rects[i][0] - 10), recognized_character, fill="red")
        img.show()

    if __name__ == '__main__':
        main()
