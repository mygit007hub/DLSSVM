import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
import random
import matplotlib.patches as patches



def load_image(frame_id):
    filename = "../Basketball/img/%04d.jpg" % frame_id
    img = io.imread(filename)
    return img

def show_img(img,border):
    fig = plt.figure()
    rect = fig.add_subplot(111, aspect='equal')
    rect.add_patch(
        patches.Rectangle(
            (border[0],border[1]),  # (x,y)
            border[2],      # width
            border[3],      # height
            fill=False,     # remove background
            edgecolor="red" # edgecolor = "#0000ff"
        )
    )
    plt.imshow(img)
    plt.show()
    return 

a = load_image(1)
show_img(a,[190,214,40,81])