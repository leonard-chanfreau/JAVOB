import os
from utils import load_image_data
from VO import VO
import numpy as np

if __name__ == '__main__':
    directory = "../data_KITTI/00/image_0"
    image_files = load_image_data(directory=directory)

    calib_file = "../data_KITTI/00/calib.txt"
    vo = VO(calib_file=calib_file)

    # get two images
    first_frame = vo.read_image_(file=os.path.join(directory, image_files[0]))
    image = vo.read_image_(file=os.path.join(directory, image_files[10]))

    # set a keyframe
    # vo.set_keyframe_(first_frame)
    # vo.keyframe_features = vo.extract_features(first_frame)

    for idx, file in enumerate(image_files):
        print(idx)

        if idx == 48:
            b=2

        image = vo.read_image_(file=os.path.join(directory, file))
        vo.run(image=image)

    # vo.initialize_point_cloud(image=image)

    a = 2