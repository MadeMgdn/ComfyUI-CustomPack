import os
import subprocess
import torch
import numpy as np

categoryFolder = "Custom pack/Video"

#Crop video to mulitple Xpx
class CP_CropVideoToME:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("IMAGE",),
                "side_width": (["left", "right"], {"default": "right"}),
                "side_height": (["top", "bottom"], {"default": "bottom"}),
                "divideBy": ([1, 8, 16, 64], {"default": 16}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "crop_video"
    CATEGORY = categoryFolder

    def crop_video(self, video, side_width, side_height, divideBy):
        B, H, W, C = video.shape
        k = divideBy

        target_w = (W // k) * k
        target_h = (H // k) * k

        if target_w == W and target_h == H:
            return (video,)

        crop_w = W - target_w
        crop_h = H - target_h

        if side_width == "left":
            x_start = crop_w
            x_end = W
        else:
            x_start = 0
            x_end = W - crop_w

        if side_height == "top":
            y_start = crop_h
            y_end = H
        else:
            y_start = 0
            y_end = H - crop_h

        cropped = video[:, y_start:y_end, x_start:x_end, :]

        return (cropped,)

#Extended video info
class CP_ExtendedVideoInfo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
            }
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "FLOAT", "FLOAT", "INT", "INT", "INT")
    RETURN_NAMES = ("frames", "audio", "fps", "duration", "frame_count", "width", "height")
    FUNCTION = "process"
    CATEGORY = categoryFolder

    def generate_silence(self, duration_sec=1.0, sample_rate=44100):
        samples = int(duration_sec * sample_rate)
        waveform = np.zeros((1, 1, samples), dtype=np.float32)
        waveform = torch.from_numpy(waveform)
        return {
            'waveform': waveform,
            'sample_rate': sample_rate,
        }

    def extract_audio_with_ffmpeg(self, video_input, sample_rate=44100, channels=2):
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        except (subprocess.SubprocessError, FileNotFoundError):
            print("FFmpeg not found.")
            return None
        is_bytesio = hasattr(video_input, 'read')

        cmd = [
            'ffmpeg',
            '-i', 'pipe:0' if is_bytesio else video_input,
            '-vn',
            '-acodec', 'pcm_f32le',
            '-f', 'f32le',
            '-ar', str(sample_rate),
            '-ac', str(channels),
            '-'
        ]

        try:
            if is_bytesio:
                proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if hasattr(video_input, 'getvalue'):
                    data = video_input.getvalue()
                else:
                    data = video_input.read()
                
                stdout, stderr = proc.communicate(input=data)
                if proc.returncode != 0:
                    print(f"ffmpeg error: {stderr.decode()}")
                    return None
                audio_bytes = stdout
            else:
                result = subprocess.run(cmd, capture_output=True, check=True)
                audio_bytes = result.stdout
        except subprocess.CalledProcessError as e:
            print(f"ffmpeg error {e.stderr.decode()}")
            return None
        except Exception as e:
            print(f"ffmpeg error: {e}")
            return None

        if not audio_bytes or len(audio_bytes) == 0:
            print("ffmpeg error: empty output.")
            return None

        expected_samples = len(audio_bytes) // (4 * channels)
        if expected_samples == 0:
            return None

        audio_float = np.frombuffer(audio_bytes, dtype=np.float32).reshape(-1, channels).T.copy()
        waveform = torch.from_numpy(audio_float).unsqueeze(0)

        return {
            'waveform': waveform,
            'sample_rate': sample_rate
        }

    def process(self, video):
        components = video.get_components()
        frames = components.images
        audio = None
        fps = float(components.frame_rate)

        if hasattr(frames, 'shape') and len(frames.shape) == 4:
            frame_count, height, width, channels = frames.shape
        elif isinstance(frames, (list, tuple)):
            frame_count = len(frames)
            if frame_count > 0:
                first = frames[0]
                if hasattr(first, 'shape') and len(first.shape) == 3:
                    height, width, channels = first.shape
                else:
                    height = width = 0
            else:
                height = width = 0
        else:
            frame_count = getattr(components, 'frame_count', 0)
            width = getattr(components, 'width', 0)
            height = getattr(components, 'height', 0)

        duration = frame_count / fps if fps > 0 else 0.0

        if audio is None:
            file_input = None
            try:
                file_input = video._VideoFromFile__file
            except AttributeError:
                pass

            if file_input is not None:
                audio = self.extract_audio_with_ffmpeg(file_input)
                if audio is None:
                    print("Audio is None. Generate silence.")
            else:
                print("Can't extract filename.")

        if audio is None:
            silence_duration = duration if duration > 0 else 1.0
            audio = self.generate_silence(duration_sec=silence_duration)

        return (frames, audio, fps, duration, frame_count, width, height)

#Selects the first non-empty video - TEST (cant eject video info after select)
class CP_FirstNonEmptyVideo:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "video1": ("VIDEO",),
                "video2": ("VIDEO",),
                "video3": ("VIDEO",),
                "video4": ("VIDEO",),
            },
        }

    RETURN_TYPES = ("VIDEO",)
    FUNCTION = "select_video"
    CATEGORY = categoryFolder

    def select_video(self, video1=None, video2=None, video3=None, video4=None):
        for vid in [video1, video2, video3, video4]:
            if vid is not None and isinstance(vid, torch.Tensor) and vid.nelement() > 0:
                return (vid,)
        return (torch.zeros(1, 1, 64, 64, 3),)

NODE_CLASS_MAPPINGS_VIDEO = {
    "CP CropVideoToME" : CP_CropVideoToME,
    "CP ExtendedVideoInfo": CP_ExtendedVideoInfo,
    "CP FirstNonEmptyVideo": CP_FirstNonEmptyVideo,
}

NODE_DISPLAY_NAME_MAPPINGS_VIDEO = {
    "CP CropVideoToME": "Video crop to ME",
    "CP ExtendedVideoInfo": "Video info extended",
    "CP FirstNonEmptyVideo": "First non empty video",
}
