import pygame
import numpy as np


def save_video_output(environment,
                      video_file_path: str ):

    model = environment.get_wrapper_attr("model")
    view = model.view
    view.gameplay_frames = []

    def before_stop():
        from moviepy.editor import ImageSequenceClip
        gameplay_clip = ImageSequenceClip(view.gameplay_frames, fps=60)
        episode_number = model.episode_count
        gameplay_clip.write_videofile(video_file_path.format(episode_number=episode_number))

    def on_frame(surface, _):
        frame = np.rot90(pygame.surfarray.array3d(surface))
        view.gameplay_frames.append(frame)

    view.on_frame = on_frame
    model.before_stop = before_stop
