import torch
import whisperx as wh
from datetime import datetime
from torch.amp.autocast_mode import autocast
import os

def init_model(device="cuda", device_index=0):
    """
    WhisperX modelini yükler.
    device: "cuda" veya "cpu"
    device_index: birden fazla GPU varsa, kullanılacak GPU indexi
    """
    print(f"Model {device} üzerinde, device_index={device_index} ile yükleniyor...")
    model = wh.load_model(
        "turbo", 
        device=device, 
        device_index=device_index,   # "gpu_index" yerine "device_index"
        compute_type="float16"       # Donanımınıza göre "float32" de kullanabilirsiniz
    )
    return model

def process_file_speech_to_text(audio_file_path, model):
    """
    Verilen ses dosyasını (audio_file_path) Whisper modeliyle çözümleyip
    metin çıktısını döndürür.
    """
    if not os.path.exists(audio_file_path):
        raise FileNotFoundError(f"Ses dosyası bulunamadı: {audio_file_path}")

    start_time = datetime.now()
    # Eğer CPU'da çalıştırıyorsanız autocast('cuda') yerine autocast('cpu') veya tamamen kaldırabilirsiniz.
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
    # Modeli GPU'da (cuda) veya CPU'da (cpu) çalıştırabilirsiniz
    # device_index=0 kullanarak 0. GPU'yu seçmiş oluyoruz.
    model = init_model(device="cuda", device_index=0)

    audio_file = "/home/ilk.mp3"  # Ses dosyası yolu

    try:
        text_output = process_file_speech_to_text(audio_file, model)
        print("Dönüştürülmüş Metin:")
        print(text_output)
    except FileNotFoundError as e:
        print(str(e))
    except Exception as e:
        print(f"Bir hata oluştu: {str(e)}")
