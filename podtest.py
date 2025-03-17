import os
import torch
import whisperx
import requests

# GPU'yu kontrol et
device = "cuda" if torch.cuda.is_available() else "cpu"

# Tüm kullanılabilir GPU'ları kullan
num_gpus = torch.cuda.device_count()
if num_gpus < 6:
    print(f"Uyarı: 6 GPU bekleniyor ancak sadece {num_gpus} adet bulundu.")
torch.set_num_threads(num_gpus * 2)  # CPU kullanımı da artırılabilir

# Dosya indirme fonksiyonu
def download_audio(url, save_path="audio.mp3"):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)
        print(f"Ses dosyası indirildi: {save_path}")
        return save_path
    else:
        raise Exception(f"Dosya indirilemedi, HTTP Kod: {response.status_code}")

# WhisperX transkripsiyon fonksiyonu
def transcribe_audio(file_path):
    model = whisperx.load_model("large-v2", device, compute_type="float16")  # Büyük model ve FP16 kullanımı
    print("Model yüklendi, transkripsiyon başlıyor...")

    # Ses dosyasını işle
    audio = whisperx.load_audio(file_path)
    
    # Transkripsiyon yap
    result = model.transcribe(audio)
    
    # Çıktıyı ekrana yazdır
    print("\nTranskripsiyon Sonucu:\n")
    print(result["text"])

# Kullanıcıdan link alma
audio_url = input("Ses dosyasının URL'sini girin: ")

# İşlem akışı
try:
    audio_file = download_audio(audio_url)
    transcribe_audio(audio_file)
except Exception as e:
    print(f"Hata oluştu: {e}")
