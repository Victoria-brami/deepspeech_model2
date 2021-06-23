import json
import pandas as pd
import os
from tqdm.notebook import tqdm
import sys
from pathlib import Path, PosixPath
import argparse

labels = ['Accelerating_and_revving_and_vroom', 'Accordion', 'Acoustic_guitar', 'Bark', 'Bass_drum', 'Bass_guitar',
          'Bathtub_(filling_or_washing)', 'Bicycle_bell',
          'Bus', 'Buzz', 'Car_passing_by', 'Chink_and_clink', 'Chirp_and_tweet', 'Church_bell',
          'Computer_keyboard', 'Crackle', 'Cricket', 'Cupboard_open_or_close', 'Cutlery_and_silverware',
          'Dishes_and_pots_and_pans', 'Drawer_open_or_close', 'Drip', 'Electric_guitar', 'Fill_(with_liquid)',
          'Finger_snapping', 'Frying_(food)', 'Glockenspiel', 'Gong', 'Harmonica', 'Hi-hat', 'Hiss', 'Keys_jangling',
          'Knock', 'Marimba_and_xylophone', 'Mechanical_fan', 'Meow', 'Microwave_oven', 'Motorcycle', 'Printer',
          'Purr', 'Race_car_and_auto_racing', 'Raindrop', 'Scissors', 'Shatter', 'Sink_(filling_or_washing)',
          'Skateboard',
          'Slam', 'Squeak', 'Stream', 'Strum', 'Tap', 'Tick-tock', 'Toilet_flush', 'Traffic_noise_and_roadway_noise',
          'Trickle_and_dribble', 'Water_tap_and_faucet', 'Waves_and_surf', 'Writing', 'Zipper_(clothing)', 'Animal',
          'Domestic_animals_and_pets', 'Dog', 'Yip', 'Howl', 'Bow-wow', 'Growling', 'Whimper_(dog)', 'Cat',
          'Caterwaul', 'Livestock_and_farm_animals_and_working_animals', 'Horse', 'Clip-clop', 'Neigh_and_whinny',
          'Cattle_and_bovinae', 'Moo', 'Cowbell', 'Pig', 'Oink', 'Goat', 'Bleat', 'Sheep', 'Fowl',
          'Chicken_and_rooster',
          'Cluck', 'Crowing_and_cock-a-doodle-doo', 'Turkey', 'Gobble', 'Duck', 'Quack', 'Goose', 'Honk',
          'Wild_animals',
          'Roaring_cats_(lions_and_tigers)', 'Roar', 'Bird', 'Bird_vocalization_and_bird_call_and_bird_song', 'Squawk',
          'Pigeon_and_dove', 'Coo', 'Crow', 'Caw', 'Owl', 'Hoot', 'Bird_flight_and_flapping_wings',
          'Canidae_and_dogs_and_wolves', 'Rodents_and_rats_and_mice', 'Mouse', 'Patter', 'Insect', 'Mosquito',
          'Fly_and_housefly', 'Bee_and_wasp_and_etc.', 'Frog', 'Croak', 'Snake', 'Rattle', 'Whale_vocalization',
          'Music', 'Musical_instrument', 'Plucked_string_instrument', 'Guitar', 'Banjo', 'Sitar', 'Mandolin', 'Zither',
          'Ukulele', 'Keyboard_(musical)', 'Piano', 'Electric_piano', 'Organ', 'Electronic_organ', 'Hammond_organ',
          'Synthesizer', 'Sampler', 'Harpsichord', 'Percussion', 'Drum_kit', 'Drum_machine', 'Drum', 'Snare_drum',
          'Rimshot', 'Drum_roll', 'Timpani', 'Tabla', 'Cymbal', 'Wood_block', 'Tambourine', 'Rattle_(instrument)',
          'Maraca', 'Tubular_bells', 'Mallet_percussion', 'Vibraphone', 'Steelpan', 'Orchestra', 'Brass_instrument',
          'French_horn', 'Trumpet', 'Trombone', 'Bowed_string_instrument', 'String_section', 'Violin_and_fiddle',
          'Pizzicato', 'Cello', 'Double_bass', 'Wind_instrument_and_woodwind_instrument', 'Flute', 'Saxophone',
          'Clarinet', 'Harp', 'Bell', 'Jingle_bell', 'Tuning_fork', 'Chime', 'Wind_chime',
          'Change_ringing_(campanology)',
          'Liquid', 'Splash_and_splatter', 'Slosh', 'Squish', 'Pour', 'Gush', 'Spray', 'Environmental_noise']


def create_json_file(list_of_labels, path_to_json='/home/coml/Documents/Victoria/noise_classifier/deepspeech_model2'):
    number_of_labels = len(list_of_labels)
    with open(os.path.join(path_to_json, 'labels_{}.json'.format(number_of_labels)), 'w') as json_file:
        json.dump(list_of_labels, json_file, indent=4)


# Define manifest creation function

def create_manifest_2(path_to_audioset_folder, data_type='eval', num_classes=183):
    """

    """
    output_name = 'audioset_{}_manifest_{}.json'.format(data_type, num_classes)
    data_path = os.path.abspath(Path(path_to_audioset_folder) / PosixPath(data_type))
    manifest_path = path_to_audioset_folder
    file_paths = list(Path(data_path / PosixPath('wav')).rglob(f"*.wav"))

    output_path = Path(manifest_path) / output_name
    output_path.parent.mkdir(exist_ok=True, parents=True)
    os.makedirs(os.path.join(data_path, 'txt_{}'.format(num_classes)), exist_ok=True)
    os.makedirs(os.path.join(data_path, 'wav'), exist_ok=True)

    print(file_paths)

    manifest = {
        'root_path': data_path,
        'samples': []
    }

    for wav_path in tqdm(file_paths, total=len(file_paths)):
        wav_path = wav_path.relative_to(data_path)
        sys.stdout.write(' \r WAV PATH:   {} '.format(wav_path))

        # Define path to write in annotations
        wav_file = str(wav_path).split('/')[-1]
        txt_name = PosixPath(wav_file.replace('.wav', '.txt'))
        transcript_path = data_path / PosixPath('txt_{}'.format(num_classes)) / txt_name
        new_wav_path = data_path / PosixPath('wav') / wav_file
        print('transcript path', transcript_path)
        print('wav path', new_wav_path)
        print()
        # sys.stdout.write(' \r NEW WAV PATH:   {} \n'.format(wav_file))
        # Write new data in the manifest
        manifest['samples'].append({
            'wav_path': new_wav_path.as_posix(),
            'transcript_path': transcript_path.as_posix()
        })
        # os.system('mv {} {}'.format(data_path / wav_path, data_path / PosixPath('wav')))

    output_path.write_text(json.dumps(manifest, indent=4), encoding='utf8')


def create_manifest(path_to_audioset_folder, size=None, data_type='eval', num_classes=183):
    """

    :param data_path: (str) path to the folder where all the wav files are
    :param output_name: (str) name of the manifest, that will be a json file dataset_manifest_datatype_num_classes.json
    :param manifest_path: (str) directory in which the manifest will be stored
    :param file_extension: (str) wav file per default
    :return: A json file containing for each sample the path to the txt file and the path to the wav file
    """
    manifest = {
        'root_path': data_path,
        'samples': []
    }

    if size == 'small':
        output_name = 'audioset_small_{}_manifest_{}.json'.format(data_type, num_classes)
    else:
        output_name = 'audioset_{}_manifest_{}.json'.format(data_type, num_classes)

    data_path = os.path.abspath(Path(path_to_audioset_folder) / PosixPath(data_type))
    manifest_path = path_to_audioset_folder
    file_paths = list(Path(data_path / PosixPath('txt_{}'.format(num_classes))).rglob(f"*.txt"))

    output_path = Path(manifest_path) / output_name
    output_path.parent.mkdir(exist_ok=True, parents=True)



    for txt_path in tqdm(file_paths, total=len(file_paths)):
        txt_path = txt_path.relative_to(data_path)

        # Define path to write in annotations
        txt_file = str(txt_path).split('/')[-1]
        wav_name = PosixPath(txt_file.replace('.txt', '.wav'))

        transcript_path = data_path / PosixPath('txt_{}'.format(num_classes)) / txt_file
        new_wav_path = PosixPath('/gpfsdswork/dataset/AudioSet') / PosixPath(data_type) / PosixPath(str(wav_name)[0]) / wav_name
        # sys.stdout.write(' \r TXT PATH:   {} '.format(transcript_path))
        sys.stdout.write(' \r WAV PATH:   {} '.format(new_wav_path))

        # Write new data in the manifest
        if size == 'small':
            if Path(new_wav_path).is_file():
                manifest['samples'].append({
                    'wav_path': new_wav_path.as_posix(),
                    'transcript_path': transcript_path.as_posix()
                })
        else:
            manifest['samples'].append({
                'wav_path': new_wav_path.as_posix(),
                'transcript_path': transcript_path.as_posix()
            })

    output_path.write_text(json.dumps(manifest, indent=4), encoding='utf8')



# define parsing transcript function
def _parse_labels(path_to_audioset_folder, data_type, num_classes):  # eval, unbalanced_train of balanced_train_function
    """

    :param data_path: (str) path to the directory where the csv file is stored
    :param csv_file: (str) name of the csv file
    :param list_of_labels: Corresponding labels (human filtered or not)
    :return: Creates for each wav a txt file containing the encoded label
    """
    labels_path = os.path.join(path_to_audioset_folder, 'csv', 'class_labels_indices_{}.csv'.format(num_classes))
    csv_labels = pd.read_csv(labels_path)
    num_classes = len(csv_labels)

    csv_file = '{}_segments_{}.csv'.format(data_type, num_classes)
    csv_path = os.path.join(path_to_audioset_folder, 'csv', csv_file)
    data_contents_df = pd.read_csv(csv_path)

    for index, row in tqdm(data_contents_df.iterrows(), total=data_contents_df.shape[0]):
        sys.stdout.write('\r  Parsing Labels .... [{} / {}]'.format(index, data_contents_df.shape[0]))
        # Get the labels name
        labels_list = row['translated_positive_labels'].split(',')
        labels = [0] * num_classes
        for lab in labels_list:
            # Fine the label corresponding index
            idx_series = csv_labels[csv_labels['display_name'] == lab]
            idx = idx_series['freesound_index'].item()
            labels[idx] = 1

        # Create labels file
        previous_path = PosixPath('{}'.format(data_type))
        txt_name = PosixPath('{}_{:.3f}_{:.3f}.txt'.format(row['YTID'], row['start_seconds'], row['end_seconds']))

        # create transcript path
        os.makedirs(path_to_audioset_folder / previous_path / PosixPath('txt_{}'.format(num_classes)), exist_ok=True)
        transcript_path = path_to_audioset_folder / previous_path / PosixPath('txt_{}'.format(num_classes)) / txt_name

        with open(transcript_path, 'w') as json_file:
            json.dump(str(labels), json_file)


def merge_manifests():
    return None


def build_argparse():
    parser = argparse.ArgumentParser(description='Processes and Downloads Freesound dataset')
    parser.add_argument('--data_type', type=str,
                        default='balanced_train', choices=['eval', 'unbalanced_train', 'balanced_train'])
    parser.add_argument('--path_to_audioset_folder', type=str,
                        default='/home/coml/Documents/Victoria/noise_classifier/deepspeech_model2/data/audioset')
    parser.add_argument('--path_to_csv_labels', type=str,
                        default='/home/coml/Documents/Victoria/noise_classifier/deepspeech_model2/data/audioset/csv/class_labels_indices_183.csv')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = build_argparse()
    _parse_labels(args.path_to_audioset_folder, args.path_to_csv_labels, data_type=args.data_type)
    output_name = 'audioset_{}_manifest_183.json'.format(args.data_type)
    manifest_path = args.path_to_audioset_folder
    create_manifest(args.path_to_audioset_folder, output_name, manifest_path, size='small', file_extension='wav',
                    data_type=args.data_type, num_classes=183)
    create_manifest(args.path_to_audioset_folder, output_name, manifest_path, file_extension='wav',
                    data_type=args.data_type, num_classes=183)
    create_json_file(list_of_labels=labels,
                     path_to_json='/home/coml/Documents/Victoria/noise_classifier/deepspeech_model2')
