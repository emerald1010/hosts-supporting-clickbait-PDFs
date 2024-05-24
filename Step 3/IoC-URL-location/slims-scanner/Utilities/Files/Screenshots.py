import os
import subprocess
from pathlib import Path
from PIL import Image

class PDFPPM():
    def __init__(self, filename, image_path, tag):
        self.tag = tag
        self.image_path = image_path
        self.image_root = os.path.join(image_path, tag)
        if not os.path.exists(self.image_path): os.makedirs(self.image_path)
        self.ret = subprocess.Popen(["pdftoppm", "-f", "1", "-l", "1", "-png", filename, self.image_root], stdout = subprocess.PIPE, stderr = subprocess.PIPE)
        self.output, error = self.ret.communicate()
        error = error.decode('utf-8').strip()
        if len(error) > 0:
            print(error)
        self.num_images = self.get_num_images()
        self.images = self.get_image_paths()

    def get_num_images(self):
        return len(os.listdir(self.image_path))

    def get_image_paths(self): # returns one path since we ask for 1 screenshot anyway
        return [os.path.join(self.image_path, x) for x in os.listdir(self.image_path) if x.startswith(self.tag) ]


def produce_thumbnail(input_image_path, output_image_path):

    output_image_path = Path(str(output_image_path))

    if output_image_path.exists():
        return

    if not os.path.exists(output_image_path.parent):
        os.makedirs(output_image_path.parent)

    pil_Image = Image.open(input_image_path)

    max_size = 200
    w, h = pil_Image.width, pil_Image.height
    aspect_ratio = min( max_size / w, max_size / h )
    new_size = ( w * aspect_ratio, h * aspect_ratio)
    pil_Image.thumbnail(new_size) # could apply filter to improve # no, ANTIALIAS does not exist
    pil_Image.save(output_image_path)

    pil_Image.close()
