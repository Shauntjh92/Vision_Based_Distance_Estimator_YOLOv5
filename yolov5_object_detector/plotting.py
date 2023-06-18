import matplotlib.pyplot as plt
import numpy as np
import cv2



def get_roi_pts(roi_raw, matrix_c2w):
    # (x1,y1),(x2,y2) = roi_raw
    x1,y1, x2, y2 = roi_raw
    pts_world = np.array([[x1, y1], [x1, y2], [x2, y2], [x2, y1], [x1, y1]])
    pts_cam = []
    for pt_world in pts_world:
        pt_cam = np.linalg.inv(matrix_c2w) @ np.array([[pt_world[0]], [pt_world[1]], [1]]).reshape(3)
        # print(pt_cam)
        # print(pt_cam[-1])
        pts_cam.append(pt_cam / pt_cam[-1])
    pts_cam = np.array(pts_cam)
    return pts_world, pts_cam[:, :2]


def plot_frame_one_row(img_raw, pts_roi_world,pts_roi_cam, pts_w):#, pairs):
    b, g, r = cv2.split(img_raw)  # get b,g,r
    img_raw = cv2.merge([r, g, b])  # switch it to rgb

    # plot
    fig = plt.figure(figsize=(8.77, 3.06))
    fig.subplots_adjust(left=0.08, bottom=0.15, right=0.98, top=0.90, wspace=0.3)
    fig.suptitle('%s (%s)' % ('Construction Site Layout', 'LTA Plan View'))
    


    # subplot 1 - camera view
    a = fig.add_subplot(1, 3, (1, 2))
    plt.imshow(img_raw)
    # a.plot(pts_roi_cam[:, 0], pts_roi_cam[:, 1], '--b')
    
    a.set_xlabel('x position (pixel)')
    a.set_ylabel('y position (pixel)')
    # a.set(xlim=(0, 2000), ylim=(1000, 1000))
    a.autoscale()

    # subplot 2 - bird eye view site
    a = fig.add_subplot(1, 3, 3)
    a.set_title('Plan View Site')
    # a.plot(pts_roi_world[:, 0], pts_roi_world[:, 1], '--b')
    a.plot((pts_w[:, 0], pts_w[:, 1]), 'og', alpha=0.5)
    
    
    # for pair in pairs:
    #     data = np.array([pts_w[pair[0]], pts_w[pair[1]]])
    #     a.plot(data[:, 0], data[:, 1], '-r')

    a.axis('equal')
    a.grid()
    a.set_xlabel('x position (meter)')
    a.set_ylabel('y position (meter)')
    a.set(xlim=(-10, 10), ylim=(30, 80))
    # a.autoscale()

    return fig


def plot_frame(dataset, img_raw, img_bev_bkgd_10x, pts_roi_cam, pts_roi_world, pts_w, pairs):
    b, g, r = cv2.split(img_raw)  # get b,g,r
    img_raw = cv2.merge([r, g, b])  # switch it to rgb

    b, g, r = cv2.split(img_bev_bkgd_10x)  # get b,g,r
    img_bev_bkgd_10x = cv2.merge([r, g, b])  # switch it to rgb

    if dataset == 'oxford_town':
        sub_2_lim = (300, -100, 0, 400)
        sub_3_lim = (20, -20, 0, 40)
        pts_roi_world[:, [0, 1]] = pts_roi_world[:, [1, 0]]
    elif dataset == 'mall':
        sub_2_lim = (-150, 150, 100, -200)
        sub_3_lim = (-15, 15, 10, -20)
    elif dataset == 'grand_central':
        sub_2_lim = (-100, 300, 400, -100)
        sub_3_lim = (-15, 35, 40, -10)
    else:
        raise Exception('Invalid dataset.')
    ps_w_10x = pts_w * 10

    # plot
    fig = plt.figure(figsize=(10, 10))

    # subplot 1 - camera view
    a = fig.add_subplot(2, 1, 1)
    plt.imshow(img_raw)
    a.plot(pts_roi_cam[:, 0], pts_roi_cam[:, 1], '--b')
    a.set_title('Video')

    # subplot 2 - bird eye view background
    a = fig.add_subplot(2, 2, 3)
    plt.imshow(img_bev_bkgd_10x)
    a.set_title('BEV')
    # a.plot(ps_w_10x[:, 1], ps_w_10x[:, 0], 'or', alpha=0.5)
    a.plot(ps_w_10x[:, 0], ps_w_10x[:, 1], 'or', alpha=0.5)

    a.axis('equal')
    a.grid()
    a.set_xlabel('0.1m / pixel')
    a.set_ylabel('0.1m / pixel')
    a.set(xlim=(sub_2_lim[0], sub_2_lim[1]), ylim=(sub_2_lim[2], sub_2_lim[3]))

    # subplot 3 - bird eye view social distancing
    a = fig.add_subplot(2, 2, 4)
    a.set_title('BEV - social distancing')
    # a.plot(pts_w[:, 1], pts_w[:, 0], 'or', alpha=0.5)
    a.plot(pts_w[:, 0], pts_w[:, 1], 'or', alpha=0.5)

    for pair in pairs:
        data = np.array([pts_w[pair[0]], pts_w[pair[1]]])
        # a.plot(data[:, 1], data[:, 0], '-g')
        a.plot(data[:, 0], data[:, 1], '-g')

    a.plot(pts_roi_world[:, 0], pts_roi_world[:, 1], '--b')
    a.axis('equal')
    a.grid()
    a.set_xlabel('meters')
    a.set_ylabel('meters')
    a.set(xlim=(sub_3_lim[0], sub_3_lim[1]), ylim=(sub_3_lim[2], sub_3_lim[3]))

    return fig