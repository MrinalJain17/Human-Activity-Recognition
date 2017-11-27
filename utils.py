import numpy as np
from cv2 import VideoCapture, cvtColor, COLOR_BGR2GRAY
from PIL import Image
from keras.preprocessing import image
from tqdm import tqdm


class Videos(object):

    def __init__(self, target_size=None, to_gray=True, max_frames=None,
                 extract_frames='middle', required_fps=None,
                 normalize_pixels=None):
        """
        Initializing the config variables

        Parameters:
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

            required_fps (int): Default 'None'
                Capture 'N' frame(s) per second from the video.

                Only the first 'N' frame(s) for each second in the video are captured.

            normalize_pixels (tuple/str): Default 'None'
                If 'None', the pixels will not be normalized.

                If a tuple - (New_min, New_max) is passed, Min-max Normalization will be used.

                If the value is 'z-score', then Z-score Normalization will be used.
                For each pixel p, z_score = (p - mean) / std
        """

        self.target_size = target_size
        self.to_gray = to_gray
        self.max_frames = max_frames
        self.extract_frames = extract_frames
        self.required_fps = required_fps
        self.normalize_pixels = normalize_pixels
        self.fps = None

    def read_videos(self, paths):
        """
        Parameters:
            paths (list): Required
                 A list of paths of the videos to be read

        Returns:
            Numpy.ndarray
                A 5-d tensor with shape (<No. of Videos>, <No. of frames>, <height>, <width>, <channels>)
        """

        list_of_videos = [
            self._read_video(path) for path in tqdm(paths)
        ]

        tensor = np.vstack(list_of_videos)

        if self.normalize_pixels != None:
            if (type(self.normalize_pixels) == tuple) and (len(self.normalize_pixels) == 2):
                base = self.normalize_pixels[0]
                r = self.normalize_pixels[1] - base
                min_ = np.min(tensor)
                max_ = np.max(tensor)
                return ((tensor.astype('float32') - min_) / (max_ - min_)) * r + base

            elif self.normalize_pixels == 'z-score':
                mean = np.mean(tensor)
                std = np.std(tensor)
                return (tensor.astype('float32') - mean) / std

            else:
                raise ValueError('Invalid value of \'normalize_pixels\'')

        return tensor

    def get_frame_count(self, paths):
        """
        Can be used to determine the value of `max_frames`

        Parameters:
            paths (list): Required
                 A list of paths of the videos to be read

        Returns:
            dict (python dictionary)
                For each video, the total number of frames in that video is stored in the dictionary.
        """

        frame_count = {}
        for path in paths:
            cap = VideoCapture(path)
            frame_count[path] = int(cap.get(7))
            cap.release()

        return frame_count

    def _read_video(self, path):
        """
        Parameters:
            path (str): Required
                Path of the video to be read

        Returns:
            Numpy.ndarray
                A 5-d tensor with shape (1, <No. of frames>, <height>, <width>, <channels>)
        """

        cap = VideoCapture(path)
        list_of_frames = []
        self.fps = int(cap.get(5))                  # Frame Rate

        while(cap.isOpened()):
            ret, frame = cap.read()
            capture_frame = True
            if self.required_fps != None:
                is_valid = range(1, (self.required_fps + 1))
                capture_frame = (int(cap.get(1)) % self.fps) in is_valid

            if ret:
                if capture_frame:

                    if self.target_size is not None:
                        temp_image = image.array_to_img(frame)
                        frame = image.img_to_array(
                            temp_image.resize(
                                self.target_size,
                                Image.ANTIALIAS)).astype('uint8')

                    if self.to_gray:
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
        if self.max_frames is not None:
            temp_video = self._process_video(video=temp_video)

        return np.expand_dims(temp_video, axis=0)

    def _process_video(self, video):
        """
        Parameters:
            video (Numpy.ndarray):
                Shape = (<No. of frames>, <height>, <width>, <channels>)

                Video whose frames are to be extracted

        Returns:
            Numpy.ndarray
                A tensor (processed video) with shape (<`max_frames`>, <height>, <width>, <channels>)
        """

        total_frames = video.shape[0]
        if self.max_frames <= total_frames:

            if self.extract_frames == 'first':
                video = video[:self.max_frames]
            elif self.extract_frames == 'last':
                video = video[(total_frames - self.max_frames):]
            elif self.extract_frames == 'middle':
                # No. of frames to remove from the front
                front = ((total_frames - self.max_frames) // 2) + 1
                video = video[front:(front + self.max_frames)]
            else:
                raise ValueError('Invalid value of \'extract_frames\'')

        else:
            raise IndexError(
                'Required number of frames is greater than the total number of frames in the video')

        return video
