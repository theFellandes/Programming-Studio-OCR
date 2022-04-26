from PIL import Image, ImageDraw, ImageFont, ImageOps
import numpy as np
from Methods import Label
from test import Test

class Ocr:

    def __init__(self, path):
        self.path = path

    def main(self):

        # Read characters from an image file
        img = Image.open(self.path).convert('RGB')
        img_gray = img.convert('L')  # converts the image to grayscale image
        img_gray.show()  # show the image
        ONE = 200  # set value of 1-valued pixels
        a = np.asarray(img_gray)  # convert from PIL to np array
        instance = Label(self)
        musty = Test()
        a_bin = instance.threshold(a, 150, ONE, 0)  # threshold the image a, with threshold T, LOW and HIGH
        im = Image.fromarray(a_bin)  # from np array to PIL format
        im.show()
        label = instance.blob_coloring_8_connected(a_bin, ONE)  # labels 8-connected components
        new_img2 = instance.np2PIL_color(label)  # converts from np array to PIL format
        new_img2.show()  # shows the image
        rects = instance.rectangle(label)
        fontsize = 30
        font = ImageFont.truetype("arial.ttf", fontsize)
        imgdraw = ImageDraw.Draw(img)
        for i in range(len(rects)):
            imgdraw.rectangle((rects[i][1], rects[i][0], rects[i][3], rects[i][2]), outline="red")
            cropped_image = new_img2.crop((rects[i][1] - 2, rects[i][0] - 2, rects[i][3] + 2, rects[i][2] + 2))
            recognized_character = instance.negative(cropped_image, musty=musty)
            imgdraw.text(((rects[i][1] + rects[i][3]) / 2, rects[i][0] - 10), recognized_character, fill="red")
        img.show()

    if __name__ == '__main__':
        main()
