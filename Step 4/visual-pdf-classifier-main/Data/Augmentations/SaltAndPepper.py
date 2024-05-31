import numpy as np
import PIL
import torch


class SaltAndPepperNoise(object):
    r""" Implements 'Salt-and-Pepper' noise
    Adding grain (salt and pepper) noise
    (https://en.wikipedia.org/wiki/Salt-and-pepper_noise)

    assumption: high values = white, low values = black

    Inputs:
            - threshold (float):
            - imgType (str): {"cv2","PIL"}
            - lowerValue (int): value for "pepper"
            - upperValue (int): value for "salt"
            - noiseType (str): {"SnP", "RGB"}
    Output:
            - image ({np.ndarray, PIL.Image}): image with
                                               noise added
    """

    def __init__(self,
                 treshold: float = 0.005,
                 imgType: str = "cv2",
                 lowerValue: int = 5,
                 upperValue: int = 250,
                 noiseType: str = "SnP"):
        self.treshold = treshold
        self.imgType = imgType
        self.lowerValue = lowerValue  # 255 would be too high
        self.upperValue = upperValue  # 0 would be too low
        if (noiseType != "RGB") and (noiseType != "SnP"):
            raise Exception("'noiseType' not of value {'SnP', 'RGB'}")
        else:
            self.noiseType = noiseType
        super(SaltAndPepperNoise).__init__()

    def __call__(self, img):
        if self.imgType == "PIL":
            img = np.array(img)

        if type(img) != np.ndarray:
            raise TypeError("Image is not of type 'np.ndarray'!")

        if self.noiseType == "SnP":
            random_matrix = np.random.rand(img.shape[0], img.shape[1])
            img[random_matrix >= (1 - self.treshold)] = self.upperValue
            img[random_matrix <= self.treshold] = self.lowerValue
        elif self.noiseType == "RGB":
            random_matrix = np.random.random(img.shape)
            img[random_matrix >= (1 - self.treshold)] = self.upperValue
            img[random_matrix <= self.treshold] = self.lowerValue

        if self.imgType == "cv2":
            return img
        elif self.imgType == "PIL":
            # return as PIL image for torchvision transforms compliance
            return PIL.Image.fromarray(img)