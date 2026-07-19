import torch

categoryFolder = "Custom pack/Audio"

#Create silence if empty audio
class CP_SilenceIfEmptyAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "duration": ("FLOAT", {"default": 60.00, "min": 0.00, "step": 0.01}),
                "sample_rate": ("INT", {"default": 44100, "min": 1, "max": 192000}),
                "channels": ("INT", {"default": 2, "min": 1, "max": 2}),
            },
            "optional": {
                "audio": ("AUDIO",), 
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "process"
    CATEGORY = categoryFolder

    def process(self, duration, sample_rate, channels, audio=None):
        if audio is None:
            waveform = self._generate_silence(duration, sample_rate, channels)
            return ({ "waveform": waveform, "sample_rate": sample_rate},)
        
        sr = audio.get('sample_rate')
        wav = audio.get('waveform')

        if sr is None or wav is None or wav.size == 0:
            waveform = self._generate_silence(duration, sample_rate, channels)
            return ({"waveform": waveform, "sample_rate": sample_rate},)
        return (audio,)
    
    def _generate_silence(self, duration, sample_rate, channels):
        num_samples = int(round(duration * sample_rate))
        return torch.zeros((1, channels, num_samples), dtype=torch.float32)

NODE_CLASS_MAPPINGS_AUDIO = {
    "CP SilenceIfEmptyAudio" : CP_SilenceIfEmptyAudio,
}

NODE_DISPLAY_NAME_MAPPINGS_AUDIO = {
    "CP SilenceIfEmptyAudio": "Silence if empty audio",
}
