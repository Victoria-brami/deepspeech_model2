from __future__ import unicode_literals
import youtube_dl
import os
import pandas as pd
import argparse
import sys

def download_audioset_sound_files(path_to_audioset_csv_folder, data_type='eval'):
    """

    :param path_to_audioset_csv_folder:
    :param data_type:
    :return:
    """

    csv_data = pd.read_csv(os.path.join(path_to_audioset_csv_folder, '{}_segments.csv'.format(data_type)))

    # Create the foder in which we store the sound files
    new_folder = path_to_audioset_csv_folder.split('/')[:-1]
    new_folder.append(data_type)
    sound_folder = '/'.join(new_folder)
    os.makedirs(os.path.join(sound_folder, 'wav'), exist_ok=True)

    for idx in range(len(csv_data)):

        video_id = csv_data['YTID'][idx]
        start_time = csv_data['start_seconds'][idx] + 1.0
        end_time = csv_data['end_seconds'][idx] - 1.0

        sys.stdout.write('\r  Video {}/{} information: {}  {}  {}'.format(idx, len(csv_data), video_id, start_time, end_time))
        os.system('youtube-dl "{id}" --quiet --extract-audio --audio-format wav '
                  '--output "{folder}/wav/{outname}"'.format(id='https://youtube.com/watch?v=' + video_id,
                                                         outname=video_id +'.%(ext)s',
                                                         folder=sound_folder))
        os.system('ffmpeg -loglevel quiet -i "{folder}/wav/{outname}.wav" '
                  '-ar "44100" -ss "{start}" -to "{end}" "{folder}/wav/{outname}_out.wav"'.format(outname=video_id,
                                                                                       start=start_time,
                                                                                       end=end_time,
                                                                                       folder=sound_folder))
        os.system('mv "{folder}/wav/{outname}_out.wav" "{folder}/wav/{outname}.wav"'.format(outname=video_id, folder=sound_folder))


def build_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type',
                        default='eval')
    parser.add_argument('--path_to_audioset_folder',
                        default='/home/coml/Documents/Victoria/noise_classifier/deepspeech_model2/data/audioset/csv')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = build_argparse()
    download_audioset_sound_files(args.path_to_audioset_folder, args.data_type)

#audio_downloader = youtube_dl.YoutubeDL({'format':'mp4'})
# audio_downloader.extract_info('http://www.youtube.com/watch?v=--U7joUcTCo', download=False)
"""

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

# THIS PORTION OF CODE WORKS !
# os.system('youtube-dl -x -ss {} -to {} --audio-format wav {}'.format(30.000, 40.000, 'https://youtube.com/watch?v='+'3B0WbwAkByI'))
os.system('youtube-dl "{}" --quiet --extract-audio --audio-format wav --output "{}"'.format('https://youtube.com/watch?v='+'3B0WbwAkByI', '3B0WbwAkByI.%(ext)s'))
os.system('ffmpeg -loglevel quiet -i "./3B0WbwAkByI.wav" -ar "44100" -ss "30.000" -to "40.000" "./3B0WbwAkByI_out.wav"')
os.system('mv "./3B0WbwAkByI_out.wav" "./3B0WbwAkByI.wav"')
# os.system('gzip "./3B0WbwAkByI.wav"')
"""
