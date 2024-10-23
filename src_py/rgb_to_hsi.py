import numpy as np

def _norm_img(img):
    img_norm = img.astype(np.float32) / 255.0
    return img_norm

def bgr_to_hsi(img):
    img_norm = _norm_img(img)
    img_out = np.zeros_like(img_norm, dtype=np.float32)

    B = img_norm[:, :, 0]
    G = img_norm[:, :, 1]
    R = img_norm[:, :, 2]

    M = np.maximum(np.maximum(R, G), B)
    m = np.minimum(np.minimum(R, G), B)
    c = M - m

    img_out[:, :, 2] = (R + G + B) / 3

    c_zero_mask = c != 0

    R_mask = (M == R) & c_zero_mask
    G_mask = (M == G) & c_zero_mask
    B_mask = (M == B) & c_zero_mask

    img_out[:, :, 0][R_mask] = 60 * np.mod((G[R_mask] - B[R_mask]) / c[R_mask], 6)
    img_out[:, :, 0][G_mask] = 60 * ((B[G_mask] - R[G_mask]) / c[G_mask] + 2)
    img_out[:, :, 0][B_mask] = 60 * ((R[B_mask] - G[B_mask]) / c[B_mask] + 4)

    img_out[:, :, 1][c_zero_mask] = 1 - (m[c_zero_mask] / c[c_zero_mask])

    return img_out

if __name__ == '__main__':
    import cv2
    test_img = cv2.imread('HW7-Auxilliary/data/testing/cloudy251.jpg')

    hsi = bgr_to_hsi(test_img)

    hsi_real = cv2.cvtColor(test_img, cv2.COLOR_BGR2HSV)
    #bgr = cv2.cvtColor(_img_to_byte(hsi), cv2.COLOR_HSV2BGR)

    cv2.imshow("im1", (hsi[:, :, 0] * 255 / 360.0).astype(np.uint8))
    cv2.imshow("im2", hsi_real[:, :, 0])
    print(np.unique(hsi[:, :, 0] / 360.0))
    cv2.waitKey(0)