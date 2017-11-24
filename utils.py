import numpy as np
from cv2 import VideoCapture, cvtColor, COLOR_BGR2GRAY
from PIL import Image
from keras.preprocessing import image
from keras.utils import to_categorical
from tqdm import tqdm


def _read_video(path, target_size=None, to_gray=True,
                max_frames=None, extract_frames='middle', single_fps=False):
    """
    Parameters:
        path (str): Required
            Path of the video to be read

        target_size (tuple): (New_Width, New_Height), Default 'None'
            A tuple denoting the target width and height of each frame in the video

        to_gray (boolean): Default 'True'
            If True, then each frame will be converted to gray scale. Otherwise, not.

        max_frames (int): Default 'None'
            The maximum number of frames to return. Extra frames are removed based on the value of `extract_frames`.

            If `single_fps` is set to `True`, then extra frames are removed from the total frames captured (at rate of 1 frame per second)

        extract_frames (str): {'first', 'middle', 'last'}, Default 'middle'
            'first': Extract the first 'N' frames

            'last': Extract the last 'N' frames

            'middle': Extract 'N' frames from the middle
                Remove ((total_frames - max_frames) // 2) frames from the beginning as well as the end

        single_fps (boolean): Default 'False'
            Capture 1 frame per second from the video.
            (Only the first frame for each second in the video is captured)

    Returns:
        Numpy.ndarray
            A tensor with shape (1, <No. of frames>, <height>, <width>, <channels>)
    """

    cap = VideoCapture(path)
    list_of_frames = []
    fps = int(cap.get(5))                  # Frame Rate

    while(cap.isOpened()):
        ret, frame = cap.read()
        capture_frame = True
        if single_fps:
            capture_frame = (cap.get(1) % fps) == 1

        if ret:
            if capture_frame:
                if target_size is not None:

                    temp_image = image.array_to_img(frame)
                    frame = image.img_to_array(
                        temp_image.resize(
                            target_size,
                            Image.ANTIALIAS)).astype('uint8')

                if to_gray:
                    gray = cvtColor(frame, COLOR_BGR2GRAY)

                    # Expanding dimension for gray channel
                    # Shape of each frame -> (<height>, <width>, 1)
                    list_of_frames.append(np.expand_dims(gray, axis=2))
                else:
                    # Shape of each frame -> (<height>, <width>, 3)
                    list_of_frames.append(frame)
        else:
            break

    cap.release()

    temp_video = np.stack(list_of_frames)
    if max_frames is not None:
        temp_video = _process_video(
            video=temp_video,
            max_frames=max_frames,
            extract_frames=extract_frames)

    return np.expand_dims(temp_video, axis=0)


def _process_video(video, max_frames, extract_frames='middle'):
    """
    Parameters:
        video (Numpy.ndarray):
            Shape = (<No. of frames>, <height>, <width>, <channels>)
            Video whose pixels are needed to be processed

        max_frames (int): Required
            The maximum number of frames to return. Extra frames are removed based on the value of `extract_frames`.

            If `single_fps` is set to `True`, then extra frames are removed from the total frames captured (at rate of 1 frame per second)

        extract_frames (str): {'first', 'middle', 'last'}, Default 'middle'
            'first': Extract the first 'N' frames

            'last': Extract the last 'N' frames

            'middle': Extract 'N' frames from the middle
                Remove ((total_frames - max_frames) // 2) frames from the beginning as well as the end

    Returns:
        Numpy.ndarray
            A tensor (processed video) with shape (<No. of frames>, <height>, <width>, <channels>)
    """
    total_frames = video.shape[0]
    if max_frames <= total_frames:

        if extract_frames == 'first':
            video = video[:max_frames]
        elif extract_frames == 'last':
            video = video[(total_frames - max_frames):]
        elif extract_frames == 'middle':
            # No. of frames to remove from the front
            front = ((total_frames - max_frames) // 2) + 1
            video = video[front:(front + max_frames)]
        else:
            raise ValueError('Invalid value of \'extract_frames\'')

    else:
        raise IndexError(
            'Required number of frames is greater than the total number of frames in the video')

    return video


def read_videos(paths, target_size=None, to_gray=True,
                max_frames=None, extract_frames='middle', single_fps=False, normalize_pixels=True):
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

        single_fps (boolean): Default 'False'
            Extract 1 frame per second from the videos.
            (Only the first frame for each second in the video is extracted)

        normalize_pixels (boolean): Default 'True'
            If 'True', then each pixel value will be normalized to the range [-1, 1]. Otherwise, not.

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
            extract_frames=extract_frames,
            single_fps=single_fps) for path in tqdm(paths)]

    tensor = np.vstack(list_of_videos)

    if normalize_pixels:
        return (tensor - 128).astype('float32') / 128

    return tensor


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
