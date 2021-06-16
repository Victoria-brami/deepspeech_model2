from __future__ import unicode_literals
import os
import pandas as pd
import argparse
import sys
from pathlib import Path, PosixPath


def download_audioset_sound_files(path_to_audioset_folder, data_type, num_classes):
    """

    :param path_to_audioset_csv_folder:
    :param data_type:
    :return:
    """

    csv_data = pd.read_csv(os.path.join(path_to_audioset_folder, 'csv', '{}_segments_{}.csv'.format(data_type, num_classes)))

    # Create the folder in which we store the sound files
    sound_folder = Path(path_to_audioset_folder) / PosixPath(data_type)
    os.makedirs(os.path.join(sound_folder, 'wav'), exist_ok=True)

    for idx in range(len(csv_data)):
        video_id = csv_data['YTID'][idx]
        start_time = csv_data['start_seconds'][idx] + 1.0  # remove sound files borders
        end_time = csv_data['end_seconds'][idx] - 1.0  # remove sound files borders

        sys.stdout.write(
            '\r  Video {}/{} information: {}  {}  {}'.format(idx, len(csv_data), video_id, start_time, end_time))
        os.system('youtube-dl "{id}" --quiet --extract-audio --audio-format wav '
                  '--output "{folder}/wav/{outname}"'.format(id='https://youtube.com/watch?v=' + video_id,
                                                             outname=video_id + '.%(ext)s',
                                                             folder=sound_folder))
        os.system('ffmpeg -loglevel quiet -i "{folder}/wav/{outname}.wav" '
                  '-ar "44100" -ss "{start}" -to "{end}" "{folder}/wav/{outname}_out.wav"'.format(outname=video_id,
                                                                                                  start=start_time,
                                                                                                  end=end_time,
                                                                                                  folder=sound_folder))
        os.system('mv "{folder}/wav/{outname}_out.wav" "{folder}/wav/{outname}.wav"'.format(outname=video_id,
                                                                                            folder=sound_folder))
        # os.system('gzip "{folder}/wav/{outname}.wav"'.format(outname=video_id, folder=sound_folder))

def download_audioset_sound_files_2(path_to_audioset_folder, data_type, num_classes):
    """

    :param path_to_audioset_csv_folder:
    :param data_type:
    :return:
    """

    csv_data = pd.read_csv(os.path.join(path_to_audioset_folder, 'csv', '{}_segments_{}.csv'.format(data_type, num_classes)))

    # Create the folder in which we store the sound files
    sound_folder = Path(path_to_audioset_folder) / PosixPath(data_type)
    os.makedirs(os.path.join(sound_folder, 'wav'), exist_ok=True)

    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'outtmpl': '%(title)s.%(etx)s',
        'quiet': False
    }
    import youtube_dl
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download(['https://youtube.com/watch?v=-JUBdOr8Hes'])  # Download into the current working directory


def build_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type',
                        default='eval')
    parser.add_argument('--path_to_audioset_folder',
                        default='/home/coml/Documents/Victoria/noise_classifier/deepspeech_model2/data/audioset')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = build_argparse()
    download_audioset_sound_files(args.path_to_audioset_folder, args.data_type, num_classes=183)
