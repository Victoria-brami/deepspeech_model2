import json
import pandas as pd
import os
from tqdm.notebook import tqdm
import sys
from pathlib import Path, PosixPath

labels = ['Accelerating_and_revving_and_vroom', 'Accordion', 'Acoustic_guitar', 'Bark', 'Bass_drum', 'Bass_guitar',
          'Bathtub_(filling_or_washing)', 'Bicycle_bell',
          'Bus', 'Buzz', 'Car_passing_by', 'Chink_and_clink', 'Chirp_and_tweet', 'Church_bell',
          'Computer_keyboard', 'Crackle', 'Cricket', 'Cupboard_open_or_close', 'Cutlery_and_silverware',
          'Dishes_and_pots_and_and_pans', 'Drawer_open_or_close', 'Drip', 'Electric_guitar', 'Fill_(with_liquid)',
          'Finger_snapping', 'Frying_(food)', 'Glockenspiel', 'Gong', 'Harmonica', 'Hi-hat', 'Hiss', 'Keys_jangling',
          'Knock', 'Marimba_and_xylophone', 'Mechanical_fan', 'Meow', 'Microwave_oven', 'Motorcycle', 'Printer',
          'Purr', 'Race_car_and_auto_racing', 'Raindrop', 'Scissors', 'Shatter', 'Sink_(filling_or_washing)', 'Skateboard',
          'Slam', 'Squeak', 'Stream', 'Strum', 'Tap', 'Tick-tock', 'Toilet_flush', 'Traffic_noise_and_roadway_noise',
          'Trickle_and_dribble', 'Water_tap_and_faucet', 'Waves_and_surf', 'Writing', 'Zipper_(clothing)', 'Animal',
          'Domestic_animals_and_pets', 'Dog', 'Yip', 'Howl', 'Bow-wow', 'Growling', 'Whimper_(dog)', 'Cat',
          'Caterwaul', 'Livestock_and_farm_animals_and_working_animals', 'Horse', 'Clip-clop', 'Neigh_and_whinny',
          'Cattle_and_bovinae', 'Moo', 'Cowbell', 'Pig', 'Oink', 'Goat', 'Bleat', 'Sheep', 'Fowl', 'Chicken_and_rooster',
          'Cluck', 'Crowing_and_cock-a-doodle-doo', 'Turkey', 'Gobble', 'Duck', 'Quack', 'Goose', 'Honk', 'Wild_animals',
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
          'Clarinet', 'Harp', 'Bell', 'Jingle_bell', 'Tuning_fork', 'Chime', 'Wind_chime', 'Change_ringing_(campanology)',
          'Liquid', 'Splash_and_splatter', 'Slosh', 'Squish', 'Pour', 'Gush', 'Spray', 'Environmental_noise']



def create_json_file(list_of_labels, path_to_json='/home/coml/Documents/Victoria/noise_classifier/deepspeech_model2'):
    number_of_labels = len(list_of_labels)
    with open(os.path.join(path_to_json, 'labels_{}.json'.format(number_of_labels)), 'w') as json_file:
        json.dump(list_of_labels, json_file, indent=4)



# Define manifest creation function

def create_manifest(data_path, output_name, manifest_path, file_extension='wav', num_classes=183):
    """

    :param data_path: (str) path to the folder where all the wav files are
    :param output_name: (str) name of the manifest, that will be a json file dataset_manfest_datatype_num_classes.json
    :param manifest_path: (str) directory in whish the manifest will be stored
    :param file_extension: (str) wav file per default
    :return:
    """
    data_path = os.path.abspath(data_path)
    file_paths = list(Path(data_path).rglob(f"*.{file_extension}"))

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
        txt_name = PosixPath(str(wav_file).split('.')[0] + '.txt')
        transcript_path = data_path / PosixPath('txt_{}'.format(num_classes)) / txt_name
        new_wav_path = data_path / PosixPath('wav') / wav_file
        # sys.stdout.write(' \r NEW WAV PATH:   {} \n'.format(wav_file))
        # Write new data in the manifest
        manifest['samples'].append({
            'wav_path': new_wav_path.as_posix(),
            'transcript_path': transcript_path.as_posix()
        })
        os.system('mv {} {}'.format(data_path / wav_path, data_path / PosixPath('wav')))

    output_path.write_text(json.dumps(manifest, indent=4), encoding='utf8')

# define parsing transcript function
def _parse_labels(data_path, csv_file, path_to_csv_labels,
                  path_to_wavs='/home/coml/Documents/Victoria/noise_classifier/deepspeech_model2/data/audioset',
                  data_type='test'): #eval, unbalanced_train of balanced_train_function
    """

    :param data_path: (str) path to the directory where the csv file is stored
    :param csv_file: (str) name of the csv file
    :param list_of_labels: Corresponding labels (human filtered or not)
    :return: Creates for each waw a txt file containing the encoded label
    """

    csv_path = os.path.join(data_path, csv_file)
    data_contents_df = pd.read_csv(csv_path)

    csv_labels = pd.read_csv(path_to_csv_labels)
    num_classes = len(csv_labels)

    for index, row in tqdm(data_contents_df.iterrows(), total=data_contents_df.shape[0]):
        # Get the labels name
        labels_list = row['translated_positive_labels'].split(',')
        labels = [0] * num_classes
        for lab in labels_list:
            # Fine the label corresponding index
            idx_series = csv_labels[csv_labels['display_name'] == lab]
            idx = idx_series['freesound_index'].item()
            labels[idx] = 1

        # Create labels file
        previous_path = PosixPath('audioset_{}'.format(data_type))
        # wav_path = row.fname
        wav_path = os.path.join(path_to_wavs, 'audioset_{}'.format(data_type), '{}.wav'.format(row['YTID']))
        txt_name = PosixPath(str(wav_path).replace('.wav', '.txt'))

        # create transcript path
        os.makedirs(data_path / previous_path / PosixPath('txt_{}'.format(num_classes)), exist_ok=True)
        transcript_path = data_path / previous_path / PosixPath('txt_{}'.format(num_classes)) / txt_name

        with open(transcript_path, 'w') as json_file:
            json.dump(str(labels), json_file)

