import os
import torch
import whisperx
from datetime import datetime


# WhisperX ile transkripsiyon yap (GPU destekli)
def transcribe_audio(audio_path, model_name="turbo", batch_size=32):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_gpus = torch.cuda.device_count()
    print(device)
    print(num_gpus)

    print(f"💡 Kullanılan Cihaz: {device}, GPU Sayısı: {num_gpus}")

    # WhisperX modelini yükle
    model = whisperx.load_model(model_name, device, compute_type="float16")

    # Çoklu GPU kullanımı
    if num_gpus > 1:
        model = torch.nn.DataParallel(model)

    # Ses dosyasını yükle ve işleme al
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, batch_size=batch_size)

    return result["segments"]

# Ana işlev (Tüm süreci yönetir)
def process_video(video_url):
    print("📥 YouTube videosu indiriliyor...")
    #video_path = download_youtube_video(video_url)

    print("🎵 Sesi çıkarılıyor (FFmpeg kullanılarak)...")
    audio_path = "/home/ilk.mp3"#extract_audio_ffmpeg(video_path)

    print("📝 Ses metne çevriliyor (WhisperX kullanılarak)...")
    startDate = datetime.now
    transcription = transcribe_audio(audio_path)
    resultTime = datetime.noe - startDate
    print(f"sonuc Suresi: {resultTime.total_seconds()} saniye")
    text_output = os.path.join("transcriptions", os.path.basename(audio_path).replace(".mp3", ".txt"))
    os.makedirs("transcriptions", exist_ok=True)

    with open(text_output, "w", encoding="utf-8") as f:
        for segment in transcription:
            f.write(f"{segment['start']:.2f} - {segment['end']:.2f}: {segment['text']}\n")

    print(f"✅ Transkripsiyon tamamlandı: {text_output}")

    return text_output

# Örnek Kullanım
if __name__ == "__main__":
    video1 = "https://youtu.be/inxnU4C06IQ" # 10:10
    video2 = "https://youtu.be/7G347yOxmrQ" # 14:52
    video3 = "https://youtu.be/DBfuRaDSS8g" # 23:17

    print("-"*50)
    print("Video 1 Başılıyor...")
    process_video(video1)
