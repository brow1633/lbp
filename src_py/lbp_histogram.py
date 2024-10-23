from hw7_EthanBrown import compute_lbp
from rgb_to_hsi import bgr_to_hsi
import numpy as np
import cv2
import taichi as ti

ti.init(arch=ti.gpu)

@ti.func
def popcount(x: ti.u32) -> ti.u32:
    # Use bit manipulation to count the number of 1s in the 32-bit integer
    x = x - ((x >> 1) & 0x55555555)
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
    x = (x + (x >> 4)) & 0x0F0F0F0F
    x = x + (x >> 8)
    x = x + (x >> 16)
    return x & 0x3F

@ti.func
def count_switches(pattern: ti.u32, p: ti.u8) -> ti.u8:
    shifted_pattern = (pattern >> 1) | ((pattern & 1) << (p - 1))
    
    switches = pattern ^ shifted_pattern
    
    return popcount(switches)

@ti.kernel
def calc_lbp_ti(image: ti.types.ndarray(dtype=ti.u8, ndim=2), p: ti.u8, r: ti.f32, output: ti.types.ndarray(dtype=ti.u8, ndim=2), binary_pattern: ti.types.ndarray(dtype=ti.u32, ndim=2)):
    height, width = image.shape[0], image.shape[1]
    
    r_int = ti.u8(ti.ceil(r))
    
    angle = 2.0 * np.pi / p
    for i, j in ti.ndrange((r_int, height - r_int), (r_int, width - r_int)):
        center_value = image[i, j]
        
        for k in range(p):
            del_k = r * ti.cos(angle*k)
            del_l = r * ti.sin(angle*k)

            k_float = i + del_k
            l_float = j + del_l

            k_base = int(ti.floor(k_float))
            l_base = int(ti.floor(l_float))
            
            delta_k = k_float - k_base
            delta_l = l_float - l_base
            
            # Interpolated pixel value
            image_val_at_p = ((1.0 - delta_k) * (1.0 - delta_l) * image[k_base, l_base] +
                                (1.0 - delta_k) * delta_l * image[k_base, l_base + 1] +
                                delta_k * delta_l * image[k_base + 1, l_base + 1] +
                                delta_k * (1.0 - delta_l) * image[k_base + 1, l_base])

            binary_pattern[i, j] |= (ti.u8(image_val_at_p >= center_value) << k)


        # Count the number of switches
        num_switches = count_switches(binary_pattern[i, j], p)

        # Directly assign output using a mathematical approach
        output[i, j] = ti.select(num_switches <= 2, popcount(binary_pattern[i, j]), ti.u8(p + 1))

def make_hist(lbp_result, p):
    return np.bincount(lbp_result.flatten(), minlength=p+2) #p+1 and 0

def calc_lbp(img, p, r):
    # Support single channel test images
    img_ds = cv2.resize(img, (64, 64))
    if len(img_ds.shape) == 3:
        hsi = bgr_to_hsi(img_ds)
        hue = (hsi[:, :, 0] * 255.0 / 360.0).astype(np.uint8)
    else:
        hue = img_ds.astype(np.uint32)

    width, height = hue.shape

    output = np.zeros((width, height), dtype=np.uint8)
    binary_buff = np.zeros((width, height), np.uint32)
    calc_lbp_ti(hue, p, r, output, binary_buff)


    return make_hist(output,p)[1:]

def multi_lbp(img, *params):
    if len(params) % 2 != 0:
        raise ValueError("The number of parameters must be even, with each pair representing (P, R).")
    
    results = []
    img_ds = cv2.resize(img, (64,64))

    if len(img_ds.shape) == 3:
        hsi = bgr_to_hsi(img_ds)
        hue = (hsi[:, :, 0] * 255.0 / 360.0).astype(np.uint8)
    else:
        hue = img.astype(np.uint8)

    width, height = hue.shape

    for i in range(0, len(params), 2):
        p = int(params[i])
        r = params[i+1]

        output = np.zeros((width, height), dtype=np.uint8)
        binary_buff = np.zeros((width, height), np.uint32)
        calc_lbp_ti(hue, p, r, output, binary_buff)
        hist = make_hist(output,p)[1:]
        results.append(hist / np.linalg.norm(hist))
    

    combined_result = np.concatenate(results, axis=0)
    return combined_result

def calc_lbp_rust(img, p, r):
    p = int(p)
    img_ds = cv2.resize(img, (64, 64))

    if len(img_ds.shape) == 3:
        hsi = bgr_to_hsi(img_ds)
        hue = (hsi[:, :, 0] * 255.0 / 360.0).astype(np.uint8)
    else:
        hue = img_ds.astype(np.uint8)
    output = compute_lbp(hue, p, r)

    return make_hist(output, p)[1:]

def multi_lbp_rust(img, *params):
    if len(params) % 2 != 0:
        raise ValueError("The number of parameters must be even, with each pair representing (P, R).")
    
    results = []
    img_ds = cv2.resize(img, (128,128))

    if len(img_ds.shape) == 3:
        hsi = bgr_to_hsi(img_ds)
        hue = (hsi[:, :, 0] * 255.0 / 360.0).astype(np.uint8)
    else:
        hue = img.astype(np.uint8)

    for i in range(0, len(params), 2):
        p = int(params[i])
        output = compute_lbp(hue, int(params[i]), params[i+1])
        results.append(make_hist(output, p)[1:])

    combined_result = np.concatenate(results, axis=0)
    return combined_result


if __name__ == '__main__':

    # img = cv2.imread('HW7-Auxilliary/data/testing/cloudy255.jpg')
    img = np.random.random((256, 256, 3))

    # img = np.array([[5, 0, 5, 0, 5, 0, 5, 0],
    #                 [5, 0, 5, 0, 5, 0, 5, 0],
    #                 [5, 0, 5, 0, 5, 0, 5, 0],
    #                 [5, 0, 5, 0, 5, 0, 5, 0],
    #                 [5, 0, 5, 0, 5, 0, 5, 0],
    #                 [5, 0, 5, 0, 5, 0, 5, 0],
    #                 [5, 0, 5, 0, 5, 0, 5, 0],
    #                 [5, 0, 5, 0, 5, 0, 5, 0]])
    # img = np.array([[5, 4, 2, 4, 2, 2, 4, 0],
    #                 [4, 2, 1, 2, 1, 0, 0, 2],
    #                 [2, 4, 4, 0, 4, 0, 2, 4],
    #                 [4, 1, 5, 0, 4, 0, 5, 5],
    #                 [0, 4, 4, 5, 0, 0, 3, 2],
    #                 [2, 0, 4, 3, 0, 3, 1, 2],
    #                 [5, 1, 0, 0, 5, 4, 2, 3],
    #                 [1, 0, 0, 4, 5, 5, 0, 1]])

    import time
    p = 8
    r = 1
    N = 100000

    start = time.time()
    for i in range(N):
        hist = calc_lbp(img, p, r)
    end = time.time()

    print(hist)
    print(end-start)

    start = time.time()
    for i in range(N):
        hist = calc_lbp_rust(img, p, r)
    end = time.time()

    print(hist)
    print(end-start)

