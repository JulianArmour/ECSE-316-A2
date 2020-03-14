from cmath import exp, pi

import cv2
import numpy as np
from matplotlib import colors as colors
from matplotlib import pyplot as plt


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
        # TODO
        raise Exception("mode 3 not implemented")
    elif mode_ == '4':
        # TODO
        raise Exception("mode 4 not implemented")
