import pygame
import numpy as np


def save_video_output(environment,
                      video_folder: str):

    model = environment.get_wrapper_attr("model")
    view = model.view
    view.gameplay_frames = []

    def before_stop():
        print("SAVING...")
        from moviepy.editor import ImageSequenceClip
        import os
        if view.gameplay_frames:
            gameplay_clip = ImageSequenceClip(view.gameplay_frames, fps=60)
            episode_number = model.episode_count
            gameplay_clip.write_videofile(filename=os.path.join(video_folder, f"episode_{episode_number:03}.mp4"),
                                          threads=8)
            view.gameplay_frames = []

    def on_frame(surface, _):
        frame = np.rot90(pygame.surfarray.array3d(surface))
        view.gameplay_frames.append(frame)

    view.on_frame = on_frame
    model.before_stop = before_stop
