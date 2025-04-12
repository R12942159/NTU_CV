import numpy as np


def solve_homography(u, v):
    """
    This function should return a 3-by-3 homography matrix,
    u, v are N-by-2 matrices, representing N corresponding points for v = T(u)
    :param u: N-by-2 source pixel location matrices
    :param v: N-by-2 destination pixel location matrices
    :return:
    """
    N = u.shape[0]
    H = None

    if v.shape[0] is not N:
        print('u and v should have the same size')
        return None
    if N < 4:
        print('At least 4 points should be given')

    # TODO: 1.forming A
    A = []
    for (a, b), (c, d) in zip(u, v):
        A.append([a, b, 1, 0, 0, 0, -a*c, -b*c, -c])
        A.append([0, 0, 0, a, b, 1, -a*d, -b*d, -d])
    A = np.array(A)
    
    # TODO: 2.solve H with A
    _, _, Vt = np.linalg.svd(A)
    h = Vt[-1].reshape(3, 3)
    h = h / h[2, 2]  # let h(2,2) = 1
    return h


def warping(src, dst, H, ymin, ymax, xmin, xmax, direction='b'):
    """
    Perform forward/backward warpping without for loops. i.e.
    for all pixels in src(xmin~xmax, ymin~ymax),  warp to destination
          (xmin=0,ymin=0)  source                       destination
                         |--------|              |------------------------|
                         |        |              |                        |
                         |        |     warp     |                        |
    forward warp         |        |  --------->  |                        |
                         |        |              |                        |
                         |--------|              |------------------------|
                                 (xmax=w,ymax=h)

    for all pixels in dst(xmin~xmax, ymin~ymax),  sample from source
                            source                       destination
                         |--------|              |------------------------|
                         |        |              | (xmin,ymin)            |
                         |        |     warp     |           |--|         |
    backward warp        |        |  <---------  |           |__|         |
                         |        |              |             (xmax,ymax)|
                         |--------|              |------------------------|

    :param src: source image
    :param dst: destination output image
    :param H:
    :param ymin: lower vertical bound of the destination(source, if forward warp) pixel coordinate
    :param ymax: upper vertical bound of the destination(source, if forward warp) pixel coordinate
    :param xmin: lower horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param xmax: upper horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param direction: indicates backward warping or forward warping
    :return: destination output image
    """

    h_src, w_src, ch = src.shape
    h_dst, w_dst, ch = dst.shape
    H_inv = np.linalg.inv(H)

    # TODO: 1.meshgrid the (x,y) coordinate pairs
    x_grid, y_grid = np.meshgrid(np.arange(xmin, xmax, 1), np.arange(ymin, ymax, 1), sparse = False)

    # TODO: 2.reshape the destination pixels as N x 3 homogeneous coordinate
    x_flat = x_grid.flatten()
    y_flat = y_grid.flatten()
    ones = np.ones_like(x_flat)
    src_coords = np.stack([x_flat, y_flat, ones], axis=0)

    if direction == 'b':
        # TODO: 3.apply H_inv to the destination pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)
        dst2src_coords = H_inv @ src_coords
        dst2src_coords /= dst2src_coords[-1]
        Ux, Uy = dst2src_coords[0].reshape(x_grid.shape), dst2src_coords[1].reshape(x_grid.shape)

        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of source image)
        mask = ((Ux >= 0) & (Ux < w_src-1)) & ((Uy >= 0) & (Uy < h_src-1))

        # TODO: 5.sample the source image with the masked and reshaped transformed coordinates
        Ux_mask, Uy_mask = Ux[mask], Uy[mask]

        # Bilinear interpolation
        Ux_mask_floor = Ux_mask.astype(int)
        Uy_mask_floor = Uy_mask.astype(int)
        dx = (Ux_mask - Ux_mask_floor).reshape(-1, 1)
        dy = (Uy_mask - Uy_mask_floor).reshape(-1, 1)
        src_interpol = np.zeros((h_src, w_src, ch))
        src_interpol[Uy_mask_floor, Ux_mask_floor, :] = ((1-dy) * (1-dx) * src[Uy_mask_floor, Ux_mask_floor, :]
                                                         + (1-dy) * (dx) * src[Uy_mask_floor, Ux_mask_floor+1, :]
                                                         + (dy) * (1-dx) * src[Uy_mask_floor+1, Ux_mask_floor, :]
                                                         + (dy) * (dx) * src[Uy_mask_floor+1, Ux_mask_floor+1, :])

        # TODO: 6. assign to destination image with proper masking
        dst[ymin:ymax,xmin:xmax][mask] = src_interpol[Uy_mask_floor, Ux_mask_floor]

    elif direction == 'f':
        # TODO: 3.apply H to the source pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)
        src2dst_coords = H @ src_coords
        src2dst_coords /= src2dst_coords[-1]
        src2dst_coords = src2dst_coords.astype(int)
        Vx, Vy = src2dst_coords[0].reshape(x_grid.shape), src2dst_coords[1].reshape(x_grid.shape)

        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of destination image)
        mask = ((Vx < w_dst) & (0 <= Vx)) & ((Vy < h_dst) & (0 <= Vy))

        # TODO: 5.filter the valid coordinates using previous obtained mask
        Vx_mask, Vy_mask = Vx[mask], Vy[mask]

        # TODO: 6. assign to destination image using advanced array indicing
        dst[Vy_mask, Vx_mask, :] = src[mask]

    return dst 