import cv2
import numpy as np
import matplotlib.pyplot as plt


def main():

    meter_per_pixel = 20.7 / 111

    #pts_image = np.float32([[557, 470], [557, 1372], [936, 1], [936, 1913]])
    tl = (500, 600)
    bl = (100, 1800)
    tr = (8300, 600)
    br = (17800, 1800)
    # tl = (250, 600)
    # bl = (50, 1800)
    # tr = (4200, 600)
    # br = (9000, 1800)
    # pts1 = [tl, bl, tr, br]
    # tl = (1500, 600)
    # bl = (1100, 2000)
    # tr = (4300, 600)
    # br = (3700, 2000)
    pts_image = np.float32([[tl], [bl], [tr], [br]])


    pts_world = np.float32([[0, 0], [0,379], [902,0], [902,379]]) * meter_per_pixel


    pts_world_10x = pts_world * 10

    matrix_cam2world = cv2.getPerspectiveTransform(pts_image, pts_world)
    matrix_cam2world10x = cv2.getPerspectiveTransform(pts_image, pts_world_10x)

    # save transformation matrix
    np.savetxt('grand_central_matrix_cam2world.txt', matrix_cam2world)
    print('Matrix Saved.')

    # background in bev
    img = cv2.imread('grand_central_background.png')
    dst = cv2.warpPerspective(img, matrix_cam2world10x, (400, 400))
    plt.subplot(121), plt.imshow(img), plt.title('Input')
    plt.subplot(122), plt.imshow(dst), plt.title('Output')
    plt.show()
    cv2.imwrite('grand_central_background_calibrated.png', dst)
    print('Converted background saved.')



if __name__ == '__main__':
    main()

