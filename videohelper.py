import cv2

def video_frame_generator(filename):

    """
    A generator function that returns a frame on each 'next()' call.
    """
    video = cv2.VideoCapture(filename)

    while video.isOpened():
        ret, frame = video.read()

        if ret:
            yield frame
        else:
            break

    video.release()
    yield None


def mp4_video_writer(filename, frame_size, fps=20):
    """
    Opens and returns a video for writing.
    """
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    return cv2.VideoWriter(filename, fourcc, fps, frame_size)
