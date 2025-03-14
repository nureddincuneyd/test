import subprocess
import os

def convert_mp4_to_mp3(input_file, output_file="output_audio.mp3"):
    """
    FFmpeg kullanarak .mp4 formatındaki bir dosyayı .mp3 dosyasına dönüştürür.

    Parametreler
    -----------
    input_file : str
        Dönüştürülecek .mp4 dosyasının tam yol veya relatif yol adresi.
    output_file : str
        Oluşturulacak .mp3 dosyasının adı (varsayılan: output_audio.mp3).

    Dönüş Değeri
    ------------
    str
        Dönüştürülmüş .mp3 dosyasının tam yolu.
    """
    try:
        # Not: Windows'ta ffmpeg'i PATH'e eklemezseniz, tam yolu yazmanız gerekir:
        # örn. "C:/ffmpeg/bin/ffmpeg.exe"
        ffmpeg_cmd = [
            "ffmpeg",
            "-i", input_file,        # Giriş dosyası
            "-vn",                   # Video track'i devre dışı bırak (sadece ses)
            "-ab", "192k",           # Bitrate örnek, 192 kbps
            "-ar", "44100",          # Örnekleme hızı, isterseniz 48000 yapabilirsiniz
            "-y",                    # Üzerine yazmak için sorusuz kabul et
            output_file
        ]

        subprocess.run(ffmpeg_cmd, check=True)
        abs_out = os.path.abspath(output_file)
        print(f"MP3 dönüştürme işlemi tamamlandı: {abs_out}")
        return abs_out

    except subprocess.CalledProcessError as e:
        print("FFmpeg dönüşüm sırasında bir hata oluştu:", e)
        return None

if __name__ == "__main__":
    mp4_input = "ornek_video.mp4"          # Buraya kendi MP4 dosya yolunuzu yazın
    mp3_output = "ornek_ses.mp3"           # Çıkış MP3 dosya adını yazın
    convert_mp4_to_mp3(mp4_input, mp3_output)
