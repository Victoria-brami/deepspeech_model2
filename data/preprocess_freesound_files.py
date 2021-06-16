import argparse
import json
import os
import sys
from pathlib import Path, PosixPath
import pandas as pd
from tqdm.notebook import tqdm
from urllib.request import urlopen
from zipfile import ZipFile
import requests
import wget

LIST_OF_LABELS = ['Accelerating_and_revving_and_vroom', 'Accordion', 'Acoustic_guitar', 'Bark', 'Bass_drum',
                  'Bass_guitar',
                  'Bathtub_(filling_or_washing)', 'Bicycle_bell',
                  'Bus', 'Buzz', 'Car_passing_by', 'Chink_and_clink', 'Chirp_and_tweet', 'Church_bell',
                  'Computer_keyboard', 'Crackle', 'Cricket', 'Cupboard_open_or_close', 'Cutlery_and_silverware',
                  'Dishes_and_pots_and_and_pans', 'Drawer_open_or_close', 'Drip', 'Electric_guitar',
                  'Fill_(with_liquid)',
                  'Finger_snapping', 'Frying_(food)', 'Glockenspiel', 'Gong', 'Harmonica', 'Hi-hat', 'Hiss',
                  'Keys_jangling',
                  'Knock', 'Marimba_and_xylophone', 'Mechanical_fan', 'Meow', 'Microwave_oven', 'Motorcycle', 'Printer',
                  'Purr', 'Race_car_and_auto_racing', 'Raindrop', 'Scissors', 'Shatter', 'Sink_(filling_or_washing)',
                  'Skateboard',
                  'Slam', 'Squeak', 'Stream', 'Strum', 'Tap', 'Tick-tock', 'Toilet_flush',
                  'Traffic_noise_and_roadway_noise',
                  'Trickle_and_dribble', 'Water_tap_and_faucet', 'Waves_and_surf', 'Writing', 'Zipper_(clothing)',
                  'Animal',
                  'Domestic_animals_and_pets', 'Dog', 'Yip', 'Howl', 'Bow-wow', 'Growling', 'Whimper_(dog)', 'Cat',
                  'Caterwaul', 'Livestock_and_farm_animals_and_working_animals', 'Horse', 'Clip-clop',
                  'Neigh_and_whinny',
                  'Cattle_and_bovinae', 'Moo', 'Cowbell', 'Pig', 'Oink', 'Goat', 'Bleat', 'Sheep', 'Fowl',
                  'Chicken_and_rooster',
                  'Cluck', 'Crowing_and_cock-a-doodle-doo', 'Turkey', 'Gobble', 'Duck', 'Quack', 'Goose', 'Honk',
                  'Wild_animals',
                  'Roaring_cats_(lions_and_tigers)', 'Roar', 'Bird', 'Bird_vocalization_and_bird_call_and_bird_song',
                  'Squawk',
                  'Pigeon_and_dove', 'Coo', 'Crow', 'Caw', 'Owl', 'Hoot', 'Bird_flight_and_flapping_wings',
                  'Canidae_and_dogs_and_wolves', 'Rodents_and_rats_and_mice', 'Mouse', 'Patter', 'Insect', 'Mosquito',
                  'Fly_and_housefly', 'Bee_and_wasp_and_etc.', 'Frog', 'Croak', 'Snake', 'Rattle', 'Whale_vocalization',
                  'Music', 'Musical_instrument', 'Plucked_string_instrument', 'Guitar', 'Banjo', 'Sitar', 'Mandolin',
                  'Zither',
                  'Ukulele', 'Keyboard_(musical)', 'Piano', 'Electric_piano', 'Organ', 'Electronic_organ',
                  'Hammond_organ',
                  'Synthesizer', 'Sampler', 'Harpsichord', 'Percussion', 'Drum_kit', 'Drum_machine', 'Drum',
                  'Snare_drum',
                  'Rimshot', 'Drum_roll', 'Timpani', 'Tabla', 'Cymbal', 'Wood_block', 'Tambourine',
                  'Rattle_(instrument)',
                  'Maraca', 'Tubular_bells', 'Mallet_percussion', 'Vibraphone', 'Steelpan', 'Orchestra',
                  'Brass_instrument',
                  'French_horn', 'Trumpet', 'Trombone', 'Bowed_string_instrument', 'String_section',
                  'Violin_and_fiddle',
                  'Pizzicato', 'Cello', 'Double_bass', 'Wind_instrument_and_woodwind_instrument', 'Flute', 'Saxophone',
                  'Clarinet', 'Harp', 'Bell', 'Jingle_bell', 'Tuning_fork', 'Chime', 'Wind_chime',
                  'Change_ringing_(campanology)',
                  'Liquid', 'Splash_and_splatter', 'Slosh', 'Squish', 'Pour', 'Gush', 'Spray', 'Environmental_noise']

freesound_human_labels = ['Applause', 'Burping_and_eructation', 'Cheering', 'Chewing_and_mastication',
                          'Child_speech_and_kid_speaking', 'Clapping', 'Crowd', 'Fart', 'Female_singing',
                          'Female_speech_and_woman_speaking', 'Gasp', 'Gurgling', 'Male_singing',
                          'Male_speech_and_man_speaking',
                          'Run', 'Screaming', 'Sigh', 'Sneeze', 'Walk_and_footsteps', 'Whispering', 'Yell']

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

FREESOUND_ANNS_URL = 'https://zenodo.org/record/3612637/files/FSDKaggle2019.meta.zip?download=1'


def download_freesound_dataset(path_to_freesound_folder):
    os.makedirs(path_to_freesound_folder, exist_ok=True)

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
    zf.extractall(path=path_to_freesound_folder)
    # close the ZipFile instance
    zf.close()

    # 2. Download sound folders
    for data_type in FREESOUND_WAV_URLS.keys():
        sys.stdout.write('\r Downloading and extracting Freesound {} set sound files ... '.format(data_type))

        # 1. Unzip sound folders
        if data_type != 'train_noisy':
            zip_file_name, zip_url = list(FREESOUND_WAV_URLS[data_type].items())[0]

            target_unpacked_dir = os.path.join(path_to_freesound_folder, 'FSDKaggle2019.audio_{}.zip'.format(data_type))

            if os.path.exists(target_unpacked_dir):
                print('Find existing folder {}'.format(target_unpacked_dir))

            else:
                print("\n Could not find Freesound {}, Downloading corpus...".format(data_type))

                filename = wget.download(zip_url, path_to_freesound_folder)
                print("FILENAME", filename)
                name_file = 'FSDKaggle2019.audio_{}.zip'.format(data_type)
                target_file = ZipFile(os.path.join(path_to_freesound_folder, name_file))

                os.makedirs(os.path.join(path_to_freesound_folder, 'FSDKaggle2019.audio_{}'.format(data_type)), exist_ok=True)
                print("Unpacking corpus to {} ...".format(os.path.join(path_to_freesound_folder, 'FSDKaggle2019.audio_{}'.format(data_type))))
                # tar = tarfile.open(target_file)
                target_file.extractall(os.path.join(path_to_freesound_folder, 'FSDKaggle2019.audio_{}'.format(data_type)))
                target_file.close()
                os.system('rm {}'.format(target_unpacked_dir))

        else:
            # Merge in a zipped folder
            os.makedirs(os.path.join(path_to_freesound_folder, 'FSDKaggle2019.audio_train_noisy'), exist_ok=True)

            zip_url = FREESOUND_WAV_URLS[data_type]

            for zip_file_name, zip_file_url in zip_url.items():
                audio_file_url = requests.get(zip_file_url)
                audio_file_url_content = audio_file_url.content
                res_file = open(
                    os.path.join(path_to_freesound_folder, 'FSDKaggle2019.audio_train_noisy', zip_file_name), 'wb')
                res_file.write(audio_file_url_content)
                res_file.close()
            os.system('zip -s 0 {} --out {}'.format(
                os.path.join(path_to_freesound_folder, 'FSDKaggle2019.audio_train_noisy.zip'),
                os.path.join(path_to_freesound_folder, 'unsplit.zip')))
            os.system('unzip {} -d {}'.format(os.path.join(path_to_freesound_folder, 'unsplit.zip'),
                                              os.path.join(path_to_freesound_folder,
                                                           'FSDKaggle2019.audio_train_noisy')))
            os.system('rm {}'.format(os.path.join(path_to_freesound_folder, 'unsplit.zip')))


def filter_freesound_on_human_labels(path_to_freesound_folder, human_labels, data_type):
    csv_path = os.path.join(path_to_freesound_folder, 'csv', '{}_post_competition.csv'.format(data_type))
    csv_data = pd.read_csv(csv_path)

    filtered_anns = dict(fname=[], labels=[], license=[])
    for idx in range(len(csv_data)):
        labels_idx = set(csv_data['labels'][idx].split(','))
        if len(labels_idx.intersection(human_labels)) == 0:
            filtered_anns['fname'].append(csv_data['fname'][idx])
            filtered_anns['labels'].append(csv_data['labels'][idx])
            filtered_anns['license'].append(csv_data['license'][idx])

    filtered_anns = pd.DataFrame(filtered_anns)
    filtered_anns.to_csv(
        os.path.join(path_to_freesound_folder, 'csv', '{}_post_competition_filtered.csv'.format(data_type)),
        index=False)


def _parse_freesound_labels(path_to_freesound_folder, list_of_labels, data_type, num_classes):
    """
    
    :param path_to_freesound_folder: 
    :param list_of_labels: 
    :param data_type: 
    :param num_classes: 
    :return: 
    """""
    csv_path = os.path.join(path_to_freesound_folder, 'csv', '{}_post_competition_filtered.csv'.format(data_type))
    csv_data = pd.read_csv(csv_path)

    for index, row in tqdm(csv_data.iterrows(), total=csv_data.shape[0]):
        labels_list = row['labels'].split(',')
        labels = [0] * num_classes
        for lab in labels_list:
            idx = list_of_labels.index(lab)
            labels[idx] = 1

        # Create labels file
        previous_path = PosixPath('FSDKaggle2019.audio_{}'.format(data_type))
        wav_path = row.fname
        txt_name = PosixPath(str(wav_path).replace('.wav', '.txt'))

        os.makedirs(path_to_freesound_folder / previous_path / PosixPath('txt_{}'.format(num_classes)), exist_ok=True)
        transcript_path = path_to_freesound_folder / previous_path / PosixPath('txt_{}'.format(num_classes)) / txt_name

        with open(transcript_path, 'w') as json_file:
            json.dump(str(labels), json_file)


def create_freesound_manifest(path_to_freesound_folder, data_type, num_classes):
    """

    :param path_to_freesound_folder:
    :param data_type:
    :param num_classes:
    :return:
    """

    manifest_path = path_to_freesound_folder
    output_name = 'freesound_{}_manifest_{}.json'.format(data_type, num_classes)
    data_path = os.path.abspath(Path(path_to_freesound_folder) / PosixPath(data_type))
    file_paths = list(Path(data_path / PosixPath('wav')).rglob(f"*.wav"))

    output_path = Path(manifest_path) / output_name
    output_path.parent.mkdir(exist_ok=True, parents=True)
    os.makedirs(os.path.join(data_path, 'txt_{}'.format(num_classes)), exist_ok=True)
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
        txt_name = PosixPath(str(wav_file).replace('.wav', '.txt'))
        transcript_path = data_path / PosixPath('txt_{}'.format(num_classes)) / txt_name
        new_wav_path = data_path / PosixPath('wav') / wav_file

        # Write new data in the manifest
        manifest['samples'].append({
            'wav_path': new_wav_path.as_posix(),
            'transcript_path': transcript_path.as_posix()
        })
        os.system('mv {} {}'.format(data_path / wav_path, data_path / PosixPath('wav')))

    output_path.write_text(json.dumps(manifest, indent=4), encoding='utf8')


def BUILD_ARGPARSE():
    parser = argparse.ArgumentParser(description='Processes and Downloads Freesound dataset')
    # parser = add_data_opts(parser)
    parser.add_argument("--path_to_freesound_folder",
                        default='/gpfsscratch/rech/rnt/uuj49ar/freesound',
                        type=str,
                        help="Directory to store the dataset.")
    parser.add_argument("--num_classes", '-nb',
                        default=183,
                        help="Number of classes indices")
    parser.add_argument("--download", "-d", default=False)
    args = parser.parse_args()
    return args


def preprocess_freesound_files(args):

    if args.download:
        download_freesound_dataset(args.path_to_freesound_folder)
        os.makedirs(os.path.join(args.path_to_freesound_folder, 'csv'), exist_ok=True)
        os.system('mv {} {}'.format(os.path.join(args.path_to_freesound_folder, 'FSDKaggle2019.meta/*'), os.path.join(args.path_to_freesound_folder, 'csv')))
        print(' 1)  All annotations and sound files downloaded ! ')
        idx = 2
    else:
        idx = 1


    for data_type in ['train_curated', 'train_noisy', 'test']:
        filter_freesound_on_human_labels(args.path_to_freesound_folder, set(freesound_human_labels), data_type)
        print(' {})  {}: CSV Files rightly filtered ! '.format(idx, data_type))

        create_freesound_manifest(args.path_to_freesound_folder, data_type, args.num_classes)
        print(' {})  {}: Json manifests created ! '.format(idx + 1, data_type))

        _parse_freesound_labels(args.path_to_freesound_folder, LIST_OF_LABELS, data_type, args.num_classes)
        print(' {})  {}: All labels created ! '.format(idx + 2, data_type))

        idx += 3

if __name__ == '__main__':
    args = BUILD_ARGPARSE()
    preprocess_freesound_files(args)
