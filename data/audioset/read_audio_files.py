from __future__ import unicode_literals
import youtube_dl
import os


ydl_opts = {
    'format': 'bestaudio/best',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'mp3',
        'preferredquality': '192',
    }],
}
with youtube_dl.YoutubeDL(ydl_opts) as ydl:
    ydl.download(['http://www.youtube.com/watch?v=BaW_jenozKc'])



filename = '/home/coml/Documents/Victoria/noise_classifier/deepspeech_model2/data/audioset/audio/features/bal_train/4v.tfrecord'

os.system('youtube-dl -x --audio-format wav {}'.format('https://youtube.com/watch?v='+'3B0WbwAkByI'))