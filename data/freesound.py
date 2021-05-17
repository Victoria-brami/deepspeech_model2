import argparse
import json
import os
import sys
from pathlib import Path, PosixPath
from urllib.request import urlopen
from zipfile import ZipFile

import pandas as pd
import requests
from tqdm.notebook import tqdm

from deepspeech_pytorch.data.data_opts import add_data_opts
# from deepspeech_pytorch.data.utils import create_manifest

MIN_DURATION = 0
MAX_DURATION = 3

HUMAN_LABELS = {'Applause', 'Burping_and_eructation', 'Cheering', 'Chewing_and_mastication',
                'Child_speech_and_kid_speaking', 'Clapping', 'Crowd', 'Fart', 'Female_singing',
                'Female_speech_and_woman_speaking', 'Gasp', 'Gurgling', 'Male_singing', 'Male_speech_and_man_speaking',
                'Run', 'Screaming', 'Sigh', 'Sneeze', 'Walk_and_footsteps', 'Whispering', 'Yell'}

ALL_LABELS = {'Accelerating_and_revving_and_vroom', 'Accordion', 'Acoustic_guitar', 'Applause', 'Bark', 'Bass_drum',
              'Bass_guitar', 'Bathtub_(filling_or_washing)', 'Bicycle_bell', 'Burping_and_eructation', 'Bus', 'Buzz',
              'Car_passing_by', 'Cheering', 'Chewing_and_mastication', 'Child_speech_and_kid_speaking',
              'Chink_and_clink', 'Chirp_and_tweet', 'Church_bell', 'Clapping', 'Computer_keyboard', 'Crackle',
              'Cricket', 'Crowd', 'Cupboard_open_or_close', 'Cutlery_and_silverware', 'Dishes_and_pots_and_pans',
              'Drawer_open_or_close', 'Drip', 'Electric_guitar', 'Fart', 'Female_singing',
              'Female_speech_and_woman_speaking', 'Fill_(with_liquid)', 'Finger_snapping', 'Frying_(food)', 'Gasp',
              'Glockenspiel', 'Gong', 'Gurgling', 'Harmonica', 'Hi-hat', 'Hiss', 'Keys_jangling', 'Knock',
              'Male_singing', 'Male_speech_and_man_speaking', 'Marimba_and_xylophone', 'Mechanical_fan', 'Meow',
              'Microwave_oven', 'Motorcycle', 'Printer', 'Purr', 'Race_car_and_auto_racing', 'Raindrop', 'Run',
              'Scissors', 'Screaming', 'Shatter', 'Sigh', 'Sink_(filling_or_washing)', 'Skateboard', 'Slam', 'Sneeze',
              'Squeak', 'Stream', 'Strum', 'Tap', 'Tick-tock', 'Toilet_flush', 'Traffic_noise_and_roadway_noise',
              'Trickle_and_dribble', 'Walk_and_footsteps', 'Water_tap_and_faucet', 'Waves_and_surf', 'Whispering',
              'Writing', 'Yell', 'Zipper_(clothing)'}

LABELS = set.difference(ALL_LABELS, HUMAN_LABELS)

FREESOUND_WAV_URLS = {
    'test': {'audio_test.zip': 'https://zenodo.org/record/3612637/files/FSDKaggle2019.audio_test.zip?download=1'},
    'train_curated': {
        'audio_train_curated.zip': 'https://zenodo.org/record/3612637/files/FSDKaggle2019.audio_train_curated.zip?download=1'},
    'train_noisy': {
        'audio_train_noisy.z01': 'https://zenodo.org/record/3612637/files/FSDKaggle2019.audio_train_noisy.z01?download=1',
        'audio_train_noisy.z02': 'https://zenodo.org/record/3612637/files/FSDKaggle2019.audio_train_noisy.z02?download=1',
        'audio_train_noisy.z03': 'https://zenodo.org/record/3612637/files/FSDKaggle2019.audio_train_noisy.z03?download=1',
        'audio_train_noisy.z04': 'https://zenodo.org/record/3612637/files/FSDKaggle2019.audio_train_noisy.z04?download=1',
        'audio_train_noisy.z05': 'https://zenodo.org/record/3612637/files/FSDKaggle2019.audio_train_noisy.z05?download=1',
        'audio_train_noisy.z06': 'https://zenodo.org/record/3612637/files/FSDKaggle2019.audio_train_noisy.z06?download=1',
        'audio_train_noisy.zip': 'https://zenodo.org/record/3612637/files/FSDKaggle2019.audio_train_noisy.zip?download=1',
        },
}

FREESOUND_WAV_URLS = {
    'train_curated': {
        'audio_train_curated.zip': 'https://zenodo.org/record/3612637/files/FSDKaggle2019.audio_train_curated.zip?download=1'}
    #'test': {'audio_test.zip': 'https://zenodo.org/record/3612637/files/FSDKaggle2019.audio_test.zip?download=1'}
    }

FREESOUND_ANNS_URL = 'https://zenodo.org/record/3612637/files/FSDKaggle2019.meta.zip?download=1'


def BUILD_ARGPARSE():
    parser = argparse.ArgumentParser(description='Processes and Downloads Freesound dataset')
    parser = add_data_opts(parser)
    parser.add_argument("--target-dir", default='Freesound_dataset/', type=str, help="Directory to store the dataset.")
    parser.add_argument("--filter_human_sounds", default='Freesound_dataset/', type=str,
                        help="Directory to store the dataset.")
    args = parser.parse_args()
    return args


""" Start downloading files, then treat then in a second step to fit the right shape"""


def download_freesound_dataset(target_dir):
    os.makedirs(target_dir, exist_ok=True)

    # 1. Download annotations
    sys.stdout.write('\r Downloading and extracting Freesound annotation files ... \n')
    anns_zip_url = FREESOUND_ANNS_URL
    zipresp = urlopen(anns_zip_url)
    anns_tempzip = open("/tmp/anns_tempfile.zip", "wb")
    # Write the contents of the downloaded file into the new file
    anns_tempzip.write(zipresp.read())
    # Close the newly-created file
    anns_tempzip.close()
    # Re-open the newly-created file with ZipFile()
    zf = ZipFile("/tmp/anns_tempfile.zip")
    # note that extractall will automatically create the path
    zf.extractall(path=target_dir)
    # close the ZipFile instance
    zf.close()

    # 2. Download sound folders
    for data_type in FREESOUND_WAV_URLS.keys():
        sys.stdout.write('\r Downloading and extracting Freesound {} set sound files ... '.format(data_type))

        zip_file_name, zip_url = list(FREESOUND_WAV_URLS[data_type].items())[0]

        # 1. Unzip sound folders
        if data_type != 'noisy':
            """
            zipresp = urlopen(zip_url)
            tempzip = open("/tmp/tempfile.zip", "wb")

            # Write the contents of the downloaded file into the new file
            tempzip.write(zipresp.read())

            # Close the newly-created file
            tempzip.close()

            # Re-open the newly-created file with ZipFile()
            zf = ZipFile("/tmp/tempfile.zip")

            # note that extractall will automatically create the path
            zf.extractall(path=target_dir)

            # close the ZipFile instance
            zf.close()
            """
            audio_file_url = requests.get(zip_url)
            audio_file_url_content = audio_file_url.content
            res_file = open(os.path.join(target_dir, zip_file_name), 'wb')
            res_file.write(audio_file_url_content)
            res_file.close()
            os.system('unzip {} -d {}'.format(os.path.join(target_dir, zip_file_name), os.path.join(target_dir)))
            os.system('rm {}'.format(os.path.join(target_dir, zip_file_name)))

        else:
            # Merge in a zipped folder
            os.makedirs(os.path.join(target_dir, 'FSDKaggle2019.audio_train_noisy'), exist_ok=True)
            for zip_file_name, zip_file_url in zip_url.items():
                audio_file_url = requests.get(zip_file_url)
                audio_file_url_content = audio_file_url.content
                res_file = open(os.path.join(target_dir, 'FSDKaggle2019.audio_train_noisy', zip_file_name), 'wb')
                res_file.write(audio_file_url_content)
                res_file.close()
            os.system('zip -s 0 {} --out {}'.format(os.path.join(target_dir, 'FSDKaggle2019.audio_train_noisy.zip'),
                                                    os.path.join(target_dir, 'unsplit.zip')))
            os.system('unzip {} -d {}'.format(os.path.join(target_dir, 'unsplit.zip'),
                                              os.path.join(target_dir, 'FSDKaggle2019.audio_train_noisy')))
            os.system('rm {}'.format(os.path.join(target_dir, 'unsplit.zip')))


""" generate labelled non human sounds """


def remove_human_labels(target_dir):
    for data_type in FREESOUND_WAV_URLS.keys():
        anns = pd.read_csv(os.path.join(target_dir, 'FSDKaggle2019.meta/{}_post_competition.csv'.format(data_type)))
        print(anns.keys())

        filtered_anns = dict(fname=[], labels=[], license=[])
        for idx in range(len(anns)):
            labels_idx = set(anns['labels'][idx].split(','))
            if len(labels_idx.intersection(HUMAN_LABELS)) == 0:
                filtered_anns['fname'].append(anns['fname'][idx])
                filtered_anns['labels'].append(anns['labels'][idx])
                filtered_anns['license'].append(anns['license'][idx])

        filtered_anns = pd.DataFrame(filtered_anns)
        filtered_anns.to_csv(
            os.path.join(target_dir, 'FSDKaggle2019.meta/{}_post_competition_filtered.csv'.format(data_type)),
            index=False)

    return None


def create_manifest(data_path, output_name, manifest_path, file_extension='wav'):
    data_path = os.path.abspath(data_path)
    file_paths = list(Path(data_path).rglob(f"*.{file_extension}"))

    output_path = Path(manifest_path) / output_name
    output_path.parent.mkdir(exist_ok=True, parents=True)
    os.makedirs(os.path.join(data_path, 'txt'), exist_ok=True)
    os.makedirs(os.path.join(data_path, 'wav'), exist_ok=True)


    manifest = {
        'root_path': data_path,
        'samples': []
    }

    for wav_path in tqdm(file_paths, total=len(file_paths)):
        wav_path = wav_path.relative_to(data_path)
        sys.stdout.write(' \r WAV PATH:   {} '.format(wav_path))

        # Define path to write in annotations
        wav_file = str(wav_path).split('/')[0]
        txt_name = PosixPath(str(wav_file).split('.')[0] + '.txt')
        transcript_path = data_path / PosixPath('txt') / txt_name
        new_wav_path = data_path / PosixPath('wav') / wav_file
        # sys.stdout.write(' \r NEW WAV PATH:   {} \n'.format(wav_file))
        # Write new data in the manifest
        manifest['samples'].append({
            'wav_path': new_wav_path.as_posix(),
            'transcript_path': transcript_path.as_posix()
        })
        os.system('mv {} {}'.format(data_path / wav_path, data_path / PosixPath('wav')))

    output_path.write_text(json.dumps(manifest, indent=4), encoding='utf8')


def create_filtered_manifest(data_type, data_path, output_name, manifest_path, file_extension='wav'):
    data_path = os.path.abspath(data_path)
    file_paths = list(Path(data_path).rglob(f"*.{file_extension}"))

    output_path = Path(manifest_path) / output_name
    output_path.parent.mkdir(exist_ok=True, parents=True)
    os.makedirs(os.path.join(data_path, 'txt_filtered'), exist_ok=True)
    os.makedirs(os.path.join(data_path, 'wav'), exist_ok=True)

    filtered_data = pd.read_csv(os.path.join(manifest_path, 'FSDKaggle2019.meta', '{}_post_competition_filtered.csv'.format(data_type)))

    manifest = {
        'root_path': data_path,
        'samples': []
    }

    for idx, row  in tqdm(filtered_data.iterrows(), total=filtered_data.shape[0]):

        wav_path = row.fname
        sys.stdout.write(' \r WAV PATH:   {}'.format(wav_path))

        # Define path to write in annotations
        txt_name = PosixPath(str(wav_path).split('.')[0] + '.txt')
        transcript_path = data_path / PosixPath('txt_filtered') / txt_name
        new_wav_path = data_path / PosixPath('wav') / wav_path

        # Write new data in the manifest
        manifest['samples'].append({
            'wav_path': new_wav_path.as_posix(),
            'transcript_path': transcript_path.as_posix()
        })

    output_path.write_text(json.dumps(manifest, indent=4), encoding='utf8')


def _parse_labels(data_path, csv_file, list_of_labels=None, data_type='test'):
    """

    :param data_path: (str) path to the directory where the csv file is stored
    :param csv_file: (str) name of the csv file
    :param list_of_labels: Corresponding labels (human filtered or not)
    :return: Creates for each waw a txt file containing the encoded label
    """

    if list_of_labels is None:
        if 'filtered' in csv_file:
            list_of_labels = LABELS
        else:
            list_of_labels = ALL_LABELS

    csv_path = os.path.join(data_path, csv_file)
    data_contents_df = pd.read_csv(csv_path)
    num_classes = len(list_of_labels)

    for index, row in tqdm(data_contents_df.iterrows(), total=data_contents_df.shape[0]):
        labels_list = row['labels'].split(',')
        labels = [0] * num_classes
        for lab in labels_list:
            idx = sorted(list(list_of_labels)).index(lab)
            labels[idx] = 1

        # Create labels file
        previous_path = PosixPath('FSDKaggle2019.audio_{}'.format(data_type))
        wav_path = row.fname
        txt_name = PosixPath(str(wav_path).split('.')[0] + '.txt')
        if num_classes < 80:
            transcript_path = data_path / previous_path / PosixPath('txt') / txt_name
            transcript_path = data_path / previous_path / PosixPath('txt_filtered') / txt_name
        else:
            os.makedirs(data_path / previous_path / PosixPath('non_filtered_txt'), exist_ok=True)
            transcript_path = data_path / previous_path / PosixPath('non_filtered_txt') / txt_name
        with open(transcript_path, 'w') as json_file:
            json.dump(str(labels), json_file)


def main():
    args = BUILD_ARGPARSE()
    target_dir = args.target_dir
    # Download dataset
    # download_freesound_dataset(target_dir)

    # Create filtered annotations
    remove_human_labels(target_dir)

    # Create manifest (filtered dataset)
    for data_type in FREESOUND_WAV_URLS:
        create_manifest(data_path=os.path.join(target_dir, 'FSDKaggle2019.audio_{}'.format(data_type)),
                        output_name='freesound_{}_manifest.json'.format(data_type), manifest_path=target_dir)
        create_filtered_manifest(data_type=data_type, data_path=os.path.join(target_dir, 'FSDKaggle2019.audio_{}'.format(data_type)),
                        output_name='freesound_{}_manifest_filtered.json'.format(data_type), manifest_path=target_dir)

        _parse_labels(data_path=os.path.join(target_dir, ),
                      csv_file='FSDKaggle2019.meta/{}_post_competition_filtered.csv'.format(data_type),
                      list_of_labels=LABELS, data_type=data_type)
        _parse_labels(data_path=os.path.join(target_dir, ),
                      csv_file='FSDKaggle2019.meta/{}_post_competition.csv'.format(data_type),
                      list_of_labels=ALL_LABELS, data_type=data_type)

    # Create manifest for filtered dataset

    print('DOWNLOAD COMPLETED !')


if __name__ == '__main__':
    main()
