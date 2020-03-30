# -*- coding: utf-8 -*-
"""
Run this script to save MINIST data set(binary file) to img file
and save labels
"""

#Import 200 pics
from PIL import Image
from common.path import MINIST_TEST_DATA, MINIST_TEST_LABEL, \
    MINIST_TRAIN_DATA, MINIST_TRAIN_LABLE, MINIST_TEST_IMG, \
    MINIST_TEST_LABELDIR, MINIST_TRAIN_IMG, MINIST_TRAIN_LABELDIR
import struct
from xlwings import xrange


# read_image and save to png
def read_image(filename):
    f = open(filename, 'rb')

    index = 0
    buf = f.read()

    f.close()

    magic, images, rows, columns = struct.unpack_from('>IIII', buf, index)
    index += struct.calcsize('>IIII')

    for i in xrange(0,3000):

        image = Image.new('L', (columns, rows))

        for x in xrange(rows):
            for y in xrange(columns):
                image.putpixel((y, x), int(struct.unpack_from('>B', buf, index)[0]))
                index += struct.calcsize('>B')

        print('save ' + str(i) + 'image')
        if 'test' in filename:
            image.save(MINIST_TEST_IMG + str(i) + '.png')
        else:
            image.save(MINIST_TRAIN_IMG + str(i) + '.png')


def read_label(filename, saveFilename):
    f = open(filename, 'rb')
    index = 0
    buf = f.read()

    f.close()

    magic, labels = struct.unpack_from('>II', buf, index)
    index += struct.calcsize('>II')

    labelArr = [0] * labels

    for x in xrange(0,3000):

        labelArr[x] = int(struct.unpack_from('>B', buf, index)[0])
        index += struct.calcsize('>B')

    save = open(saveFilename, 'w')

    save.write(','.join(map(lambda x: str(x), labelArr)))
    save.write('\n')

    save.close()
    print('save labels success')


if __name__ == '__main__':
    #read_image(MINIST_TEST_DATA)
    #read_label(MINIST_TEST_LABEL, MINIST_TEST_LABELDIR+'test_label.txt')
    read_image(MINIST_TRAIN_DATA)
    read_label(MINIST_TRAIN_LABLE, MINIST_TRAIN_LABELDIR+'train_label.txt')
