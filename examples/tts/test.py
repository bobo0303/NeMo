import os
import torch
import soundfile as sf  

from io import BytesIO

from nemo_text_processing.text_normalization.normalize import Normalizer
from nemo.collections.tts.models.base import SpectrogramGenerator, Vocoder
from nemo.collections.tts.models import Tacotron2Model, HifiGanModel, WaveGlowModel


# 檢查 CUDA 可用性並設置設備
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 創建一個文本輸入框供用戶輸入要轉換的文本
input_text = "Hey, this is a test of the speech synthesis system."

# 加載預訓練的 TTS 模型和正規化器
normalizer = Normalizer(input_case="cased", lang="en")

spectrogram_model = SpectrogramGenerator.from_pretrained(model_name="tts_en_tacotron2")
vocoder_model = Vocoder.from_pretrained(model_name="tts_en_hifigan")
# vocoder_model = Vocoder.from_pretrained(model_name="tts_en_waveglow_88m")

"""
spectrogram_model = Tacotron2Model.from_pretrained("tts_en_tacotron2").eval().to(device)
vocoder_model = HifiGanModel.from_pretrained("tts_en_hifigan").eval().to(device)
vocoder_model = WaveGlowModel.from_pretrained("tts_en_waveglow_88m").eval().to(device)
"""
if input_text:
    # 正規化文本
    normalized_text = normalizer.normalize(input_text)

    # 轉換文本為音素
    tokens = spectrogram_model.parse(normalized_text, normalize=True)

    # 生成頻譜圖
    spectrogram = spectrogram_model.generate_spectrogram(tokens=tokens)
    
    # 轉換頻譜圖為 NumPy 數據
    spectrogram_np = spectrogram.squeeze().cpu().detach().numpy()

    # 將頻譜圖轉換為音頻樣本
    audio = vocoder_model.convert_spectrogram_to_audio(spec=spectrogram)
    

    save_path = "audio/"
    if not os.path.exists(save_path):  
        os.makedirs(save_path)  
        
    audio_numpy = audio.squeeze().cpu().detach().numpy()
    audio_buffer = BytesIO()

    # 使用 soundfile 將音頻數據寫入 BytesIO 緩衝區  
    sf.write(audio_buffer, audio_numpy, samplerate=22050, format='WAV')  
      
    # 將 BytesIO 緩衝區的指針移到開頭  
    audio_buffer.seek(0)  
      
    # 將緩衝區中的音頻數據保存為文件  
    with open(os.path.join(save_path, f"hifigan - {input_text}.wav"), 'wb') as f:  
        f.write(audio_buffer.read())  
