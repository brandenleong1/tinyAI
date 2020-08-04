from PIL import Image
import numpy as np
import os


def check(test, array):
    return test in array


def convert(folder, itercount):
    for file in os.listdir(folder + '/' + str(itercount)):
        img_depict = int(file[0])
        print([file], [img_depict])

        img = Image.open(folder + '/' + str(itercount) + '/' + file).convert('LA')
        img.save('greyscale.png')

        img_g = Image.open('greyscale.png', 'r')
        pix_val = list(img_g.getdata())
        pix_val_flat = np.array([], dtype='float')

        for i in pix_val:
            pix_val_flat = np.append(pix_val_flat, float(round((i[0] / 255) * -1 + 1)))

        if (not os.path.isfile('./img_save_in_' + str(itercount) + '.npy')) and (not os.path.isfile('./img_save_out_' + str(itercount) + '.npy')):
            inputs = np.array([pix_val_flat])
            outputs = np.array([[img_depict]])

        else:
            inputs = np.load('./img_save_in_' + str(itercount) + '.npy')
            outputs = np.load('./img_save_out_' + str(itercount) + '.npy')
            if not check(pix_val_flat.tolist(), inputs.tolist()):
                inputs = np.append(inputs, [pix_val_flat], axis=0)
                outputs = np.append(outputs, [[img_depict]], axis=0)

        np.save('img_save_in_' + str(itercount), inputs)
        np.save('img_save_out_' + str(itercount), outputs)


def compress(folder, output):
    for file in os.listdir(folder):
        np.save(output, np.append(np.load(output + '.npy'), np.load(folder + '/' + file), axis=0))
    return file


# ============================================================================================

# for i in range(0, 1):  # TODO
#     convert('./mnist/testSample', i)  # TODO

# print(compress('./numpy_saves/in', 'img_save_in_0'))  # TODO
# print(compress('./numpy_saves/out', 'img_save_out_0'))  # TODO

imgconvert_inputs = np.load('img_save_in_0.npy')  # TODO
imgconvert_outputs = np.load('img_save_out_0.npy')  # TODO
print(imgconvert_inputs.shape, imgconvert_outputs.shape)
