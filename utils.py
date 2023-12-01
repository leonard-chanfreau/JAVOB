import os

def load_image_data(directory: str, filetype: str = 'png'):
    """

    Parameters
    ----------
    directory: str
        directory containing images

    filetype: str
        image type

    Returns
    -------
    image_files: list[str]
        names of the images in given directory
    """

    if not os.path.exists(directory):
        raise ValueError('directory does not exist')

    image_files = os.listdir(directory)
    image_files = [x for x in image_files if x.endswith(filetype)]

    return image_files
