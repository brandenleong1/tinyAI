from PIL import Image
import numpy as np
import os


def check(test, array):
    return test in array


def convert(image, img_depict):
    img = Image.open(image).convert('LA')
    img.save('greyscale.png')

    img_g = Image.open('greyscale.png', 'r')
    pix_val = list(img_g.getdata())
    pix_val_flat = np.array([], dtype='int64')

    for i in pix_val:
        pix_val_flat = np.append(pix_val_flat, int((i[0] / 255) * -1 + 1))

    if (not os.path.isfile('./img_save_in.npy')) and (not os.path.isfile('./img_save_out.npy')):
        inputs = np.array([pix_val_flat])
        outputs = np.array([[img_depict]], dtype='int64')

    else:
        inputs = np.load('img_save_in.npy')
        outputs = np.load('img_save_out.npy')
        if not check(pix_val_flat.tolist(), inputs.tolist()):
            inputs = np.append(inputs, [pix_val_flat], axis=0)
            outputs = np.append(outputs, [[img_depict]], axis=0)

    np.save('img_save_in', inputs)
    np.save('img_save_out', outputs)

    return inputs, outputs


imgconvert_inputs, imgconvert_outputs = convert('image.png', 3)  # TODO
