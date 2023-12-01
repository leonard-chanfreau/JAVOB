import os
from utils import load_image_data
from VO import VO

if __name__ == '__main__':
    directory = "D:/Master/VAMR/kitti/05/image_0"
    image_files = load_image_data(directory=directory)

    calib_file = "D:/Master/VAMR/kitti/05/calib.txt"
    vo = VO(calib_file=calib_file)

    image = vo.read_image_(file=os.path.join(directory,image_files[0]))

    vo.extract_features(image=image, algorithm='sift')

    a = 2