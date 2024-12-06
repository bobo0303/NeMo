import os
import csv
import time
import torch
import soundfile as sf  

from io import BytesIO

from nemo_text_processing.text_normalization.normalize import Normalizer
from nemo.collections.tts.models.base import SpectrogramGenerator, Vocoder
from nemo.collections.tts.models import Tacotron2Model, HifiGanModel, WaveGlowModel, FastPitchModel

TTS = "hifigan"
language = "en" # en

save_path = "audio/"
if not os.path.exists(save_path):  
    os.makedirs(save_path)  
    
csv_file_path = save_path + f'{TTS}_{language}_.csv'

with open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:
    csv_writer = csv.writer(csv_file)   

    # 檢查 CUDA 可用性並設置設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 創建一個文本輸入框供用戶輸入要轉換的文本
    # input_text = ["Hey, this is a test of the speech synthesis system",
    #             "Hello! It's a pleasure to serve you. Regarding the University T-shirt you mentioned, the University T-shirt  for men can be found in Area I, while the University T-shirt  for women can be found in Area B. I hope this information is helpful to you and I wish you a happy shopping",
    #             "viper one heading one one zero",
    #             "Hello! It's a pleasure to serve you. Regarding the women's jumpsuit you mentioned, you can find it in area D. There are jumpsuits, jumpsuits and overalls. I hope this information is helpful to you and I wish you a happy shopping! If you have any other questions, please feel free to tell me",
    #             ]
    
    input_text = ["您好！很高興為您服務。關於您提到的大學T，男用大學T可以在I區找到，而女用大學T則在B區。希望這些資訊對您有幫助，祝您購物愉快！",
                  "您好！很高興為您服務。關於您提到的女性連身褲，您可以在D區找到。這裡有連身長褲、連身短褲和吊帶褲。希望這些資訊對您有幫助，祝您購物愉快！如果還有其他問題，隨時告訴我哦！",
                  "您好！很高興為您服務。"]

    # 加載預訓練的 TTS 模型和正規化器
    start = time.time()
    normalizer = Normalizer(input_case="cased", lang=language)
    end = time.time()
    norm_end = end - start
    
    start = time.time()
    if language == "zh":
        # spectrogram_model = SpectrogramGenerator.from_pretrained(model_name="tts_zh_fastpitch_sfspeech")
        spectrogram_model = FastPitchModel.restore_from("/mnt/weight/FastPitch.nemo")
    elif language == "en":
        spectrogram_model = SpectrogramGenerator.from_pretrained(model_name="tts_en_tacotron2")
    end = time.time()
    spectrogram_end = end - start
    
    start = time.time()
    if TTS == "hifigan" and language == "zh":
        vocoder_model = Vocoder.from_pretrained(model_name="tts_zh_hifigan_sfspeech")
    elif TTS == "hifigan":
        vocoder_model = Vocoder.from_pretrained(model_name="tts_en_hifigan")
    elif TTS == "waveglow":
        vocoder_model = Vocoder.from_pretrained(model_name="tts_en_waveglow_88m")
    end = time.time()
    vocoder_end = end - start

    """
    spectrogram_model = Tacotron2Model.from_pretrained("tts_en_tacotron2").eval().to(device)
    vocoder_model = HifiGanModel.from_pretrained("tts_en_hifigan").eval().to(device)
    vocoder_model = WaveGlowModel.from_pretrained("tts_en_waveglow_88m").eval().to(device)
    """
    
    if input_text:
        for index, text in enumerate(input_text):
            start = time.time()     
            # 正規化文本
            normalized_text = normalizer.normalize(text)

            # 轉換文本為音素
            tokens = spectrogram_model.parse(normalized_text, normalize=True)

            # 生成頻譜圖
            spectrogram = spectrogram_model.generate_spectrogram(tokens=tokens)
            
            # 轉換頻譜圖為 NumPy 數據
            spectrogram_np = spectrogram.squeeze().cpu().detach().numpy()

            # 將頻譜圖轉換為音頻樣本
            audio = vocoder_model.convert_spectrogram_to_audio(spec=spectrogram)
            end = time.time()
            ifer_end = end - start
            
            start = time.time()     
            audio_numpy = audio.squeeze().cpu().detach().numpy()
            audio_buffer = BytesIO()

            # 使用 soundfile 將音頻數據寫入 BytesIO 緩衝區  
            sf.write(audio_buffer, audio_numpy, samplerate=22050, format='WAV')  
            
            # 將 BytesIO 緩衝區的指針移到開頭  
            audio_buffer.seek(0)  
            
            # 將緩衝區中的音頻數據保存為文件  
            with open(os.path.join(save_path, f"{TTS}_{language}_{index}.wav"), 'wb') as f:  
                f.write(audio_buffer.read())  
            end = time.time()
            save_end = end - start
            csv_writer.writerow([text, ifer_end, f'save time: {save_end}'])
            
        csv_writer.writerow(['norm load time', norm_end])
        csv_writer.writerow(['spectrogram load time', spectrogram_end])
        csv_writer.writerow(['vocoder load time', vocoder_end])
        
"""
中文TTS問題:
https://github.com/NVIDIA/NeMo/issues/7389
簡單介紹:
https://blog.csdn.net/lovechris00/article/details/129279724
CKPT:
https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/tts/checkpoints.html
HiFi-GAN: 
chrome-extension://bocbaocobfecmglnmeaeppambideimao/pdf/viewer.html?file=https%3A%2F%2Farxiv.org%2Fpdf%2F2010.05646
生MEL:
chrome-extension://bocbaocobfecmglnmeaeppambideimao/pdf/viewer.html?file=https%3A%2F%2Farxiv.org%2Fpdf%2F1712.05884
code參考:
https://ai-free-startup.medium.com/win-11-%E6%9C%AC%E5%9C%B0%E7%AB%AF%E9%81%8B%E8%A1%8C%E8%AA%9E%E9%9F%B3%E7%94%9F%E6%88%90%E6%9C%8D%E5%8B%99-nemo-streamlit-624498e800f4
官方參考:
https://colab.research.google.com/github/NVIDIA/NeMo/blob/stable/tutorials/tts/NeMo_TTS_Primer.ipynb#scrollTo=q4SAerFphP8W
https://colab.research.google.com/github/NVIDIA/NeMo/blob/stable/tutorials/tts/Inference_ModelSelect.ipynb#scrollTo=tJZaIpa2gTqx
https://colab.research.google.com/github/NVIDIA/NeMo/blob/stable/tutorials/tts/Pronunciation_customization.ipynb#scrollTo=iYoNFuRvLQ8F
"""