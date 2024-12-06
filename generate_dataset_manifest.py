
import os  
import json  
import wave  
  
# 定義音檔資料夾路徑  
audio_folder = "/mnt/test_audio"  
  
# 準備 JSON 清單  
json_data = []  
  
# 遍歷音檔  
for audio_file in os.listdir(audio_folder):  
    if audio_file.endswith(".wav"):  
        filepath = os.path.join(audio_folder, audio_file)  
        transcription = audio_file[:-4].lower()
          
        # 使用 wave 模組計算音檔持續時間  
        with wave.open(filepath, 'r') as wav_file:  
            frames = wav_file.getnframes()  
            rate = wav_file.getframerate()  
            duration = frames / float(rate)  
          
        json_data.append({  
            "audio_filepath": filepath,  
            "text": transcription,  
            "duration": duration  
        })  
  
# 寫入 JSON 檔案  
with open("audio_metadata.json", "w", encoding="utf-8") as json_file:  
    json.dump(json_data, json_file, ensure_ascii=False, indent=4)  
  
print("JSON 檔案已成功生成！")  