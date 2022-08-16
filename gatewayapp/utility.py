# Copyright (C) 2020 - 2022 APC, Inc.

from PIL import Image, ImageOps
import os

def resize(im, desired_size):
    old_size = im.size  # old_size[0] is in (width, height) format
    print('old_size', old_size)

    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    print('ratio', ratio, 'new_size', new_size)
    # use thumbnail() or resize() method to resize the input image

    # thumbnail is a in-place operation

    # im.thumbnail(new_size, Image.ANTIALIAS)

    im = im.resize(new_size, Image.ANTIALIAS)
    # create a new image and paste the resized on it

    new_im = Image.new("RGB", (desired_size, desired_size))
    new_im.paste(im, ((desired_size-new_size[0])//2,
                        (desired_size-new_size[1])//2))

    return new_im

def get_prefix_topic():
    return os.environ['PREFIX_TOPIC'] + "/"

def get_base_dir():
    return os.environ['BASE_DIR'] + "/"    