import pygame
import numpy as np


def save_video_output(environment,
                      video_folder: str):

    model = environment.model
    view = model.view
    view.gameplay_frames = []

    def before_stop():
        print("SAVING...")
        import os
        if view.gameplay_frames:
            from moviepy.editor import ImageSequenceClip
            gameplay_clip = ImageSequenceClip(view.gameplay_frames, fps=60)
            gameplay_clip.write_videofile(filename=os.path.join(video_folder, f"episode_{model.episode_count:03}.mp4"),
                                          threads=12)
            view.gameplay_frames = []

    def on_frame(surface, _):
        frame = np.rot90(pygame.surfarray.array3d(surface))
        frame = np.flipud(frame)
        view.gameplay_frames.append(frame)

    view.on_frame = on_frame
    model.before_stop = before_stop
