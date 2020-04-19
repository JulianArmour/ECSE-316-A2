from cmath import exp, pi

import cv2
import numpy as np
import math
import random
import time
from matplotlib import colors as colors
from matplotlib import pyplot as plt
from copy import copy, deepcopy


def slowft(sig):
    sig_len = len(sig)
    return [sum([sig[n] * exp(-2j * pi * k * n / sig_len) for n in range(sig_len)]) for k in range(sig_len)]


def fastft(sig):
    n = len(sig)
    if n <= 1:
        return sig
    evens = fastft(sig[0::2])
    odds = fastft(sig[1::2])
    ws = np.exp(-2j * pi * np.arange(n // 2) / n)
    return np.append(evens + ws * odds, evens - ws * odds)


def ifastft(sig):
    """
    using conjugation method
    source: https://www.dsprelated.com/showarticle/800.php
    :param sig: frequency domain numpy array
    :return: time domain numpy array
    """
    return np.conjugate(fastft(np.conjugate(sig))) / len(sig)


def fastft2(sig):
    def axis_fft(a, axis):
        return np.apply_along_axis(fastft, axis, a)

    return np.apply_over_axes(axis_fft, sig, [0, 1])


def ifastft2(sig):
    def axis_ifft(a, axis):
        return np.apply_along_axis(ifastft, axis, a)

    return np.apply_over_axes(axis_ifft, sig, [0, 1])


def pow2_ceil(num):
    """
    :param num: integer to round up
    :return: num rounded up to the nearest power of 2
    """
    return int(np.exp2(np.ceil(np.log2(num))))


def pad_image(image):
    height, width = image.shape
    x_pad = pow2_ceil(width) - width
    y_pad = pow2_ceil(height) - height
    return cv2.copyMakeBorder(image, 0, y_pad, 0, x_pad, cv2.BORDER_CONSTANT, 0)


def mode1(imgpath='moonlanding.png'):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    image = cv2.imread(imgpath, 0)
    pimg = pad_image(image)  # padded image
    freq = np.fft.fft2(pimg)
    # original image
    plt.subplot(121)
    plt.imshow(pimg, cmap='gray')
    plt.title('Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122)
    # frequency domain image
    fimg = plt.imshow(np.abs(freq), norm=colors.LogNorm(vmin=1))
    plt.title('Frequencies'), plt.xticks([]), plt.yticks([])
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(fimg, cax=cax)
    plt.show()


def mode2(img_path='moonlanding.png', red=0.8):
    """
    Filters an image's frequencies using a reduction factor.
    red * height is removed from the image's frequencies, starting from
    high frequencies; similarly with width.

    :param img_path: path to an image file.
    :param red: reduction factor to apply to height and width.
    """
    image = cv2.imread(img_path, 0)
    ih, iw = image.shape
    pimg = pad_image(image)  # padded image
    freq = fastft2(pimg)
    f_rows, f_cols = freq.shape
    r, c = np.ogrid[:f_rows, :f_cols]
    fcrow = f_rows // 2  # frequency center row
    fccol = f_cols // 2  # frequency center column
    mask = (r >= int((1 - red) * fcrow)) & (r < (1 + red) * fcrow) | \
           (c >= int((1 - red) * fccol)) & (c < (1 + red) * fccol)
    freq[mask] = 0
    # freq[int(fcrow * (1 - red)): int(fcrow * (1 + red)), int(fccol * (1 - red)): int(fccol * (1 + red))] = 0
    filtimg = np.real(ifastft2(freq))[:ih, :iw]
    plt.subplot(121)
    plt.imshow(image, cmap='gray'), plt.title('Original image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122)
    plt.imshow(filtimg, cmap='gray'), plt.title('Filtered image'), plt.xticks([]), plt.yticks([])
    plt.show()
    # print number of non-zeros leftover and fraction they represent of original Fourier coefficients
    non_zeros = int(f_rows * f_cols * (4 * ((1 - red) ** 2)))
    frac_of_fourier = non_zeros / (f_rows * f_cols)
    print(f'Number of non-zeros: {non_zeros}')
    print(f'Fraction of original Fourier: {frac_of_fourier}')
    
def mode3(img_path='moonlanding.png'):
    image = cv2.imread(img_path, 0)
    ih, iw = image.shape
    pimg = pad_image(image)
    freq = fastft2(pimg)
    f_rows, f_cols = freq.shape
    #Compression 0%
    filtimg0 = np.real(ifastft2(freq))[:ih, :iw]
    #Take the complex array and make it 1D to be able to sort and get percentile
    flattend = freq.flatten()
    flattend.sort()
    # Get the value of element in the 10% place
    index10 = math.floor(flattend.size*0.1)
    index30 = math.floor(flattend.size*0.3)
    index50 = math.floor(flattend.size*0.5)
    index75 = math.floor(flattend.size*0.75)
    index95 = math.floor(flattend.size*0.95)
    # we will make a deep copy of the original frequency to be able to change the values
    toCompress10 = freq
    toCompress30 = freq
    toCompress50 = freq
    toCompress75 = freq
    toCompress95 = freq
    # If the value is under the threshold, we set it to 0
    newArray10 = np.where(toCompress10 > flattend[index10], toCompress10, 0)
    newArray30 = np.where(toCompress30 > flattend[index30], toCompress30, 0)
    newArray50 = np.where(toCompress50 > flattend[index50], toCompress50, 0)
    newArray75 = np.where(toCompress75 > flattend[index75], toCompress75, 0)
    newArray95 = np.where(toCompress95 > flattend[index95], toCompress95, 0)
    # we take the inverse to get the imge back
    filtimg10 = np.real(ifastft2(newArray10))[:ih, :iw]
    filtimg30 = np.real(ifastft2(newArray30))[:ih, :iw]
    filtimg50 = np.real(ifastft2(newArray50))[:ih, :iw]
    filtimg75 = np.real(ifastft2(newArray75))[:ih, :iw]
    filtimg95 = np.real(ifastft2(newArray95))[:ih, :iw]

    ##Display images
    plt.subplot(131)
    plt.imshow(filtimg0, cmap='gray'), plt.title('Compression 0')
    plt.subplot(132)
    plt.imshow(filtimg10, cmap='gray'), plt.title('Compression 10')
    plt.subplot(133)
    plt.imshow(filtimg30, cmap='gray'), plt.title('Compression 30')
    plt.show()
    plt.subplot(231)
    plt.imshow(filtimg50, cmap='gray'), plt.title('Compression 50')
    plt.subplot(232)
    plt.imshow(filtimg75, cmap='gray'), plt.title('Compression 75')
    plt.subplot(233)
    plt.imshow(filtimg95, cmap='gray'), plt.title('Compression 95')
    plt.show()
    
    # we calculate how many 0 in the array
    print("Non zeros with 0% compression: ",np.count_nonzero(freq))
    print("Non zeros with 10% compression: ",np.count_nonzero(newArray10))
    print("Non zeros with 30% compression: ",np.count_nonzero(newArray30))
    print("Non zeros with 50% compression: ",np.count_nonzero(newArray50))
    print("Non zeros with 75% compression: ",np.count_nonzero(newArray75))
    print("Non zeros with 95% compression: ",np.count_nonzero(newArray95))
    
def mode4():
    array1 = np.random.random(32) + np.random.random(32) * 1j
    complex1 = 32 * 32
    array2 = np.random.random(64) + np.random.random(64) * 1j
    complex2 = 64 * 64
    array3 = np.random.random(256) + np.random.random(256) * 1j
    complex3 = 256 * 256
    array4 = np.random.random(1024) + np.random.random(1024) * 1j
    complex4 = 1024 * 1024
    array5 = np.random.random(4096) + np.random.random(4096) * 1j
    complex5 = 4096 * 4096

    ##For array1
    total = 0
    list = []
    for i in range (10):
        t0 = time.time()
        slowft(array1)
        t1 = time.time()
        list.append(t1-t0)
        total = total + t1-t0    
    std1 = np.std(list)
    mean1 = total/10
    ##for array2
    total = 0
    list = []
    for i in range (10):
        t0 = time.time()
        slowft(array2)
        t1 = time.time()
        list.append(t1-t0)
        total = total + t1-t0
    std2 = np.std(list)
    mean2 = total/10
    ##for array3
    total = 0
    list = []
    for i in range (10):
        t0 = time.time()
        slowft(array3)
        t1 = time.time()
        list.append(t1-t0)
        total = total + t1-t0
    std3 = np.std(list)
    mean3 = total/10
    ##for array4
    total = 0
    list = []
    for i in range (10):
        t0 = time.time()
        slowft(array4)
        t1 = time.time()
        list.append(t1-t0)
        total = total + t1-t0
    std4 = np.std(list)
    mean4 = total/10
    ##for array5
    total = 0
    list = []
    for i in range (10):
        t0 = time.time()
        slowft(array5)
        t1 = time.time()
        list.append(t1-t0)
        total = total + t1-t0
    std5 = np.std(list)
    mean5 = total/10
    ###FASTFTF###
    ##For array6
    total = 0
    list = []
    for i in range (10):
        t0 = time.time()
        fastft(array1)
        t1 = time.time()
        list.append(t1-t0)
        total = total + t1-t0    
    std6 = np.std(list)
    mean6 = total/10
    ##for array7
    total = 0
    list = []
    for i in range (10):
        t0 = time.time()
        fastft(array2)
        t1 = time.time()
        list.append(t1-t0)
        total = total + t1-t0
    std7 = np.std(list)
    mean7 = total/10
    ##for array8
    total = 0
    list = []
    for i in range (10):
        t0 = time.time()
        fastft(array3)
        t1 = time.time()
        list.append(t1-t0)
        total = total + t1-t0
    std8 = np.std(list)
    mean8 = total/10
    ##for array9
    total = 0
    list = []
    for i in range (10):
        t0 = time.time()
        fastft(array4)
        t1 = time.time()
        list.append(t1-t0)
        total = total + t1-t0
    std9 = np.std(list)
    mean9 = total/10
    ##for array10
    total = 0
    list = []
    for i in range (10):
        t0 = time.time()
        fastft(array5)
        t1 = time.time()
        list.append(t1-t0)
        total = total + t1-t0
    std10 = np.std(list)
    mean10 = total/10
    x=[32,64,256,1024,4096]
    y=[mean1,mean2,mean3,mean4,mean5]
    z=[mean6,mean7,mean8,mean9,mean10]
    plt.errorbar(x, y, yerr=[std1*2,std2*2,std3*2,std4*2,std5*2], fmt='-o')
    plt.errorbar(x, z, yerr=[std6*2,std7*2,std8*2,std9*2,std10*2], fmt='-o')
    plt.axis([0, 5000, 0, 20])
    plt.suptitle('Algorithm Runtime vs. Problem Size', fontsize=20)
    plt.xlabel('Problem Size ^2', fontsize=18)
    plt.ylabel('Runtime (s)', fontsize=16)
    plt.show()
       


def parse_argv(argv_list):
    def parse_flags(flag_list):
        if not flag_list:
            return {}
        flg, flg_arg, *rest = flag_list
        if flg == '-m':
            return {'mode': flg_arg, **parse_flags(rest)}
        elif flg == '-i':
            return {'image': flg_arg, **parse_flags(rest)}
        else:
            return {}

    return parse_flags(argv_list[1:])


if __name__ == "__main__":
    from sys import argv

    if len(argv) % 2 != 1:
        print('Invalid number of arguments')
        exit(-1)
    args = parse_argv(argv)
    args.setdefault('mode', '1')
    args.setdefault('image', 'moonlanding.png')
    mode_ = args['mode']
    image_ = args['image']
    if mode_ == '1':
        mode1(image_)
    elif mode_ == '2':
        mode2(image_)
    elif mode_ == '3':
        mode3(image_)
    elif mode_ == '4':
        mode4()
