import os
import torch
import ffmpeg
import whisperx
from pytube import YouTube

# YouTube videosunu indir
def download_youtube_video(video_url, output_path="downloads"):
    os.makedirs(output_path, exist_ok=True)
    yt = YouTube(video_url)
    video_stream = yt.streams.filter(file_extension="mp4", progressive=True).first()
    video_path = os.path.join(output_path, yt.title + ".mp4")
    video_stream.download(output_path, filename=yt.title + ".mp4")
    return video_path

# Videodan sesi çıkar ve MP3 olarak kaydet (FFmpeg Kullanarak)
def extract_audio_ffmpeg(video_path, output_path="audio"):
    os.makedirs(output_path, exist_ok=True)
    audio_path = os.path.join(output_path, os.path.basename(video_path).replace(".mp4", ".mp3"))

    # FFmpeg ile ses dosyasını çıkar
    ffmpeg.input(video_path).output(audio_path, format="mp3", acodec="libmp3lame").run(overwrite_output=True)
    
    return audio_path

# WhisperX ile transkripsiyon yap (GPU destekli)
def transcribe_audio(audio_path, model_name="large-v2", batch_size=16):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_gpus = torch.cuda.device_count()

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
    video_path = download_youtube_video(video_url)

    print("🎵 Sesi çıkarılıyor (FFmpeg kullanılarak)...")
    audio_path = extract_audio_ffmpeg(video_path)

    print("📝 Ses metne çevriliyor (WhisperX kullanılarak)...")
    transcription = transcribe_audio(audio_path)

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

    print("-"*50)
    print("Video 2 Başılıyor...")
    process_video(video2)

    print("-"*50)
    print("Video 3 Başılıyor...")
    process_video(video3)