import os
from utils import load_image_data
from VO import VO

if __name__ == '__main__':
    directory = "../kitti/05/image_0"
    image_files = load_image_data(directory=directory)

    calib_file = "../kitti/05/calib.txt"
    vo = VO(calib_file=calib_file)

    # get two images
    first_frame = vo.read_image_(file=os.path.join(directory, image_files[0]))
    image = vo.read_image_(file=os.path.join(directory, image_files[10]))

    # set a keyframe
    # vo.set_keyframe_(first_frame)
    # vo.keyframe_features = vo.extract_features(first_frame)

    for idx, file in enumerate(image_files):
        print(idx)
        image = vo.read_image_(file=os.path.join(directory, file))
        vo.run(image=image)

    vo.initialize_point_cloud(image=image)

    a = 2