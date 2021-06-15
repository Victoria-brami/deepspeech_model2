from __future__ import unicode_literals
import os
import pandas as pd
import argparse
import sys
from pathlib import Path, PosixPath


def download_audioset_sound_files(path_to_audioset_folder, data_type='eval'):
    """

    :param path_to_audioset_csv_folder:
    :param data_type:
    :return:
    """

    csv_data = pd.read_csv(os.path.join(path_to_audioset_folder, 'csv', '{}_segments.csv'.format(data_type)))

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
