import numpy as np
from cv2 import VideoCapture, cvtColor, COLOR_BGR2GRAY, destroyAllWindows
from PIL import Image
from keras.preprocessing import image
from keras.utils import to_categorical
from tqdm import tqdm


def _read_video(path, target_size=None, to_gray=True,
                max_frames=None, extract_frames='middle', normalize_pixels=False):
    """
    Parameters:
        path (str): Required
            Path of the video to be read

        target_size (tuple): (New_Width, New_Height), Default 'None'
            A tuple denoting the target width and height of each frame in the video

        to_gray (boolean): Default 'True'
            If True, then each frame will be converted to gray scale. Otherwise, not.

        max_frames (int): Default 'None'
            The maximum number of frames to return. Extra frames are removed based on the value of 'extract_frames'.

        extract_frames (str): {'first', 'middle', 'last'}, Default 'middle'
            'first': Extract the first 'N' frames

            'last': Extract the last 'N' frames

            'middle': Extract 'N' frames from the middle
                Remove ((total_frames - max_frames) // 2) frames from the beginning as well as the end

        normalize_pixels (boolean): Default 'True'
            If 'True', then each pixel value will be normalized (divided by 255). Otherwise, not.

    Returns:
        Numpy.ndarray
            A tensor with shape (1, <No. of frames>, <height>, <width>, <channels>)
    """

    cap = VideoCapture(path)
    list_of_frames = []

    while(cap.isOpened()):
        ret, frame = cap.read()

        if ret:
            if target_size is not None:

                temp_image = image.array_to_img(frame)
                frame = image.img_to_array(
                    temp_image.resize(
                        target_size,
                        Image.ANTIALIAS)).astype('uint8')

            if to_gray:
                # Shape of each frame after conversion to gray scale -> (<height>, <width>)
                gray = cvtColor(frame, COLOR_BGR2GRAY)

                # Expanding dimension for gray channel
                list_of_frames.append(np.expand_dims(gray, axis=2))
            else:
                # Shape of each frame -> (<height>, <width>, 3)
                list_of_frames.append(frame)
        else:
            break

    cap.release()
    destroyAllWindows()

    temp_video = np.stack(list_of_frames)

    if max_frames is not None:

        total_frames = temp_video.shape[0]
        if max_frames <= total_frames:

            if extract_frames == 'first':
                temp_video = temp_video[:max_frames]
            elif extract_frames == 'last':
                temp_video = temp_video[(total_frames - max_frames):]
            elif extract_frames == 'middle':
                # No. of frames to remove from the front
                front = ((total_frames - max_frames) // 2) + 1
                temp_video = temp_video[front:(front + max_frames)]
            else:
                raise ValueError('Invalid value of \'extract_frames\'')

        else:
            raise IndexError(
                'Required number of frames is greater than the total number of frames in the video')

    if normalize_pixels:
        return np.expand_dims((temp_video.astype('float32') / 255), axis=0)
    else:
        return np.expand_dims(temp_video, axis=0)


def read_videos(paths, target_size=None, to_gray=True,
                max_frames=None, extract_frames='middle', normalize_pixels=True):
    """
    Parameters:
        paths (list): Required
             A list of paths of the videos to be read

        target_size (tuple): (New_Width, New_Height), Default 'None'
            A tuple denoting the target width and height of each frame in each of the video

        to_gray (boolean): Default 'True'
            If True, then each frame will be converted to gray scale. Otherwise, not.

        max_frames (int): Default 'None'
            The maximum number of frames to return for each video.
            Extra frames are removed based on the value of 'extract_frames'.

        extract_frames (str): {'first', 'middle', 'last'}, Default 'middle'
            'first': Extract the first 'N' frames

            'last': Extract the last 'N' frames

            'middle': Extract 'N' frames from the middle
                Remove ((total_frames - max_frames) // 2) frames from the beginning as well as the end

        normalize_pixels (boolean): Default 'True'
            If 'True', then each pixel value will be normalized (divided by 255). Otherwise, not.

    Returns:
        Numpy.ndarray
            A tensor with shape (<No. of Videos>, <No. of frames>, <height>, <width>, <channels>)
    """

    list_of_videos = [
        _read_video(
            path,
            target_size=target_size,
            to_gray=to_gray,
            max_frames=max_frames,
            extract_frames=extract_frames) for path in tqdm(paths)]

    if normalize_pixels:
        return np.vstack(list_of_videos).astype('float32') / 255
    else:
        return np.vstack(list_of_videos)


def one_hot_encode(y, num_classes):
    """
    Parameters:
        y (Numpy.ndarray): shape - (<No. of samples>, )
            A Numpy array to be one hot encoded

        num_classes (int): Should be greater than 0
            Total number of distinct classes

    Returns:
        Numpy.ndarray
            Shape - (<No. of samples>, num_classes)
    """

    return to_categorical(y, num_classes)
