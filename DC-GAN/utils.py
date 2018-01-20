import glob
import numpy as np
import scipy.misc
import matplotlib
import matplotlib.pyplot as plt

from PIL import Image

class CelebData(object):
    def __init__(self, path):

        self.path = path
        self.files = glob.glob(path +"\\Img\\img_align_celeba\\img_align_celeba\\"+ "*.jpg")

        # Preprocessing CelebA Image
        print("Preprocessing starts")
        self.preprocess()
        print("Preprocessing is done. Total number of images:", len(self.imgs))

    def read_attr(self):
        path = self.path + "\\Anno\\list_attr_celeba.txt"
        self.attr2id = {}
        self.id2attr = {}
        i = 0
        with open(path, mode='r') as f:
            contents = f.readlines()
            feature_set = contents[1].split()

            for line in contents[2:]:
                id, feature = line.split(maxsplit=1)
                print(id)

    def preprocess(self):
        self.imgs = []
        i = 0
        for file in self.files:
            if i == 5000:
                break
            i += 1

            img = Image.open(file)
            width, height = img.size

            c_width = int(width / 2)
            c_height = int(height / 2)

            # 218*178 -> 178*178
            img = img.crop((0, c_height - 89, width, c_height + 89))
            img = img.resize((128, 128))
            img = (np.array(img)/127.5) - 1.0
            self.imgs.append(img)

            # plt.imshow(img)
            # Image._show(Image.fromarray(img))
            # matplotlib.image.imsave('name1.png')
            # break


ds = CelebData(path="D:\\yeachan\\information\\dataset\\CelebA")