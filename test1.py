import torch
import whisperx as wh
from datetime import datetime
from torch.amp.autocast_mode import autocast
import os

def init_model(gpu_id=0):
    """
    Belirtilen GPU'ya (ya da eğer CUDA yoksa CPU'ya) Whisper modelini yükler.
    Varsayılan olarak GPU 0'ı kullanır.
    """
    device = f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'
    print(f"Model {device} üzerinde yükleniyor...")
    model = wh.load_model("turbo", device=device)
    return model

def process_file_speech_to_text(audio_file_path, model):
    """
    Verilen ses dosyasını (audio_file_path) Whisper modeliyle çözümleyip
    metin çıktısını döndürür.
    """
    if not os.path.exists(audio_file_path):
        raise FileNotFoundError(f"Ses dosyası bulunamadı: {audio_file_path}")

    start_time = datetime.now()
    with autocast('cuda'), torch.no_grad():
        result = model.transcribe(audio_file_path)
    total_time = datetime.now() - start_time

    # Dil kodunu ISO formatına göre ayarlamak isterseniz:
    langCode = result.get('language', '')
    if langCode == "en":
        langCode = "en-US"
    elif langCode == "tr":
        langCode = "tr-TR"
    elif langCode == "fr":
        langCode = "fr-FR"
    elif langCode == "de":
        langCode = "de-DE"

    print(f"İşlem süresi: {total_time.total_seconds()} saniye")
    print(f"Tahmin edilen dil: {langCode}")
    return result['text']

if __name__ == '__main__':
    # GPU veya CPU üzerinde modelimizi başlatıyoruz
    model = init_model(gpu_id=0)  # GPU ID'yi değiştirebilirsiniz

    # Metne dönüştürmek istediğiniz ses dosyasının yolunu burada belirtiyoruz
    audio_file = "/home/ilk.mp3"  # Örnek isim

    # Speech-to-text işlemi
    try:
        text_output = process_file_speech_to_text(audio_file, model)
        print("Dönüştürülmüş Metin:")
        print(text_output)
    except FileNotFoundError as e:
        print(str(e))
    except Exception as e:
        print(f"Bir hata oluştu: {str(e)}")
