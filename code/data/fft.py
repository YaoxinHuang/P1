import numpy as np


def amp_spectrum_swap(amp_local, amp_target, alpha: float = 0, lamda: float = 0, axes=(0, 1)):
    """
    Swap the amplitude spectrum of two images in the frequency domain, focusing on a central low-frequency region.

    Parameters:
    - amp_local (ndarray): The amplitude spectrum of the local image.
    - amp_target (ndarray): The amplitude spectrum of the target image to swap with.
    - alpha (float): Proportion of the image dimensions to define the size of the central frequency region to swap.
    - lamda (float): Blend ratio, determines how much of the target amplitude is used in the swap.

    Returns:
    - a_local (ndarray): The modified amplitude spectrum of the local image after swapping.
    """
    a_local = np.fft.fftshift(amp_local, axes=axes)
    a_trg = np.fft.fftshift(amp_target, axes=axes)
    h, w = a_local.shape
    b = int(np.floor(np.amin((h, w)) * alpha))
    c_h = int(np.floor(h / 2.0))
    c_w = int(np.floor(w / 2.0))

    h1 = c_h - b
    h2 = c_h + b + 1
    w1 = c_w - b
    w2 = c_w + b + 1

    a_local[..., h1:h2, w1:w2] = a_local[..., h1:h2, w1:w2] * lamda + a_trg[..., h1:h2, w1:w2] * (1 - lamda)
    a_local = np.fft.ifftshift(a_local, axes=axes)
    return a_local

def domain_shift(img, target_soruce_img, axes=(0, 1)):
    img_fft = np.fft.fft2(img, axes=axes)
    target_soruce_img_fft = np.fft.fft2(target_soruce_img, axes=axes)
    amp_local = np.abs(img_fft)
    amp_target = np.abs(target_soruce_img_fft)
    new_amp_local = amp_spectrum_swap(amp_local, amp_target, alpha=0.5, lamda=0.5)
    phase_local = np.angle(img_fft)
    img_fft = new_amp_local * np.exp(1j * phase_local)
    new_img = np.fft.ifft2(img_fft, axes=axes).real

    return new_img

if __name__ == '__main__':
    new_img = domain_shift(np.random.rand(3, 1, 256, 256), np.random.rand(1, 1, 256, 256))
    print(new_img.shape)