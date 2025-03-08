from moviepy.editor import VideoFileClip

def mp4_to_gif(input_path, output_path, start_time=None, end_time=None, fps=10):
    """
    Convert an MP4 file to a GIF.
    
    :param input_path: Path to the input MP4 file.
    :param output_path: Path to save the output GIF file.
    :param start_time: Start time in seconds (optional).
    :param end_time: End time in seconds (optional).
    :param fps: Frames per second for the GIF (default: 10).
    """
    clip = VideoFileClip(input_path)
    if start_time is not None and end_time is not None:
        clip = clip.subclip(start_time, end_time)
    clip = clip.set_fps(fps)
    clip.write_gif(output_path, fps=fps)
    
if __name__ == "__main__":
    input_mp4 = "guide_dog.mp4"  # 修改为你的MP4文件路径
    output_gif = "guide_dog.gif"  # 修改为你想保存的GIF文件路径
    mp4_to_gif(input_mp4, output_gif, start_time=0, end_time=20, fps=10)