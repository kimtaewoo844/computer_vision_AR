# -*- coding: utf-8 -*-

import cv2 as cv
import numpy as np


def main(filename, K, distortion, output_filename='output.mp4'):
    board_cellsize = 25
    board_pattern = (8, 6)

    # Open a video
    video = cv.VideoCapture(filename)

    # Video info for writer
    width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv.CAP_PROP_FPS)

    # Prepare output video writer
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter(output_filename, fourcc, fps, (width, height))
    if not out.isOpened():
        raise

    # Prepare a 3D box for simple AR
    box1 = board_cellsize * np.array([[2, 2, -1], [3, 2, -1], [3, 3, -1], [2.5, 3, -1], [2.5, 2.5, -1], [2, 2.5, -1]], np.float32)
    box2 = board_cellsize * np.array([[3, 1, -1], [4, 1, -1], [4, 2, -1], [3.7, 2, -1], [3.5, 1.8, -1], [3.3, 2, -1], [3, 2, -1]], np.float32)
    box3 = board_cellsize * np.array([[4, 2, -1], [5, 2, -1], [5, 2.3, -1], [4.5, 2.3, -1], [4.5, 2.6, -1], [5, 2.6, -1], [5, 3, -1], [4, 3, -1]], np.float32)
    logo = [(box1, (255, 0, 0)), (box2, (0, 0, 255)), (box3, (0, 255, 0))]

    # Prepare 3D points on a chessboard
    col, row = board_pattern
    obj_points = board_cellsize * np.array([[c, r, 0] for r in range(row) for c in range(col)], np.float32)

    # Run pose estimation
    while True:
        # Read an image from the video
        valid, img = video.read()
        if not valid:
            break
        # Estimate the camera pose
        complete, img_points = cv.findChessboardCorners(img, board_pattern, None)
        if complete:
            ret, rvec, tvec = cv.solvePnP(obj_points, img_points, K, distortion)

            # Draw the box on the image
            for box, color in logo:
                lines, __ = cv.projectPoints(box, rvec, tvec, K, distortion)
                cv.polylines(img, [np.int32(lines)], True, color, 4)

            R, _ = cv.Rodrigues(rvec) # Alternative) `scipy.spatial.transform.Rotation`
            p = (-R.T @ tvec).flatten()

            info = f'XYZ: [{p[0]:.3f} {p[1]:.3f} {p[2]:.3f}]'
            cv.putText(img, info, (10, 25), cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0))

        out.write(img)
    cv.destroyAllWindows()
    video.release()
    out.release()


if __name__ == '__main__':
    filename = r'D:\전공\컴퓨터비전\과제\checkerboard.mp4'
    K = np.array([[2.01711732e+03, 0.00000000e+00, 9.87146503e+02],
                  [0.00000000e+00, 2.02773308e+03, 5.46443142e+02],
                  [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

    distortion = np.array([[ 4.88316744e-01, -4.59649302e+00,  2.46373398e-03, -9.35495518e-04,
                            1.27197058e+01]])

    main(filename, K, distortion)