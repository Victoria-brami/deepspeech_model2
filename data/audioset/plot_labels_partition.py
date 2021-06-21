from plotly.subplots import make_subplots
import plotly.graph_objects as go
import os
from pathlib import Path, PosixPath
import argparse
from tqdm.notebook import tqdm
import numpy as np
import json
import pandas as pd

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


def parse_transcript(transcript_path):
    with open(transcript_path, 'r', encoding='utf8') as transcript_file:
        # transcript = transcript_file.read().replace('\n', '')
        transcript = transcript_file.read().replace('\n', '')
    # transcript = list(filter(None, [self.labels_map.get(x) for x in list(transcript)]))
    new_transcript = []
    for x in transcript:
        if x not in ['', ' ', ',', ']', '[', '"']:
            new_transcript.append(int(x))
    return new_transcript

def find_music_labelled_elements():
    return None


def plot_all_labels_proportion_partition(data_path, labels=labels, data_type='train', mode='count', create_figure=False):

    num_classes = len(labels)
    all_labels = np.zeros(num_classes)

    # Load the manifest depending on the type of set
    data_filename = os.path.join(data_path, '{data}_manifest_{num}.json'.format(data=data_type, num=num_classes))
    with open(data_filename, 'r') as json_file:
        json_samples = json.load(json_file)['samples']

    for sample in json_samples:
        txt_labels = parse_transcript(sample['transcript_path'])
        all_labels += np.array(txt_labels)
    print(all_labels)

    if mode == 'percentage':
        all_labels = all_labels / np.sum(all_labels) * 100
        print('percentage', all_labels.sum())
    else:
        all_labels = [str(all_labels[i]) for i in range(len(all_labels))]
        print(all_labels)

    csv_data = dict()
    for i in range(num_classes):
        csv_data[labels[i]] = int(all_labels[i])
    csv_data = pd.DataFrame(csv_data)
    csv_data.to_csv(data_path, 'number_of_samples_per_label_183.csv', index=False)

    # Create partition figure
    if create_figure:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=labels, y=all_labels, histfunc="sum"))

        fig.update_xaxes(title_text="Labels Names")

        if mode == 'percentage':
            fig.update_yaxes(title_text="Percentage", range=[0, 8])
        elif mode == 'count':
            fig.update_yaxes(title_text="Number of samples", range=[0, 100000])

        fig.update_layout(title_text="AudioSet labels partition ({} mode)".format(mode),
                           title_font_size=30)
        fig.write_image("graphs/{}_SET_labels_{}_partition_mode_{}.png".format(data_type.upper(), num_classes, mode), width=2000, height=1000)
        fig.show()

def plot_all_labels_proportion_partition_bis(data_path, labels=labels, data_type='eval', mode='count'):

    num_classes = len(labels)
    if data_type == 'train':
        file_paths = list(
            Path(data_path / PosixPath('balanced_train') / PosixPath('txt_{}'.format(num_classes))).rglob(f"*.txt"))
        file_paths.extend(list(
            Path(data_path / PosixPath('unbalanced_train') / PosixPath('txt_{}'.format(num_classes))).rglob(f"*.txt")))
    else:
        file_paths = list(Path(data_path / PosixPath(data_type) / PosixPath('txt_{}'.format(num_classes))).rglob(f"*.txt"))
    print(file_paths)
    print(len(file_paths))
    all_labels = np.zeros(num_classes)
    for txt_path in tqdm(file_paths, total=len(file_paths)):
        txt_labels = parse_transcript(txt_path)
        all_labels += np.array(txt_labels)
    print(all_labels)

    if mode == 'percentage':
        all_labels = all_labels / np.sum(all_labels) * 100
        print('percentage', all_labels.sum())
    all_labels = [str(all_labels[i]) for i in range(len(all_labels))]
    print(all_labels)
    # Create partition figure
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=labels, y=all_labels, histfunc="sum"))

    fig.update_xaxes(title_text="Labels Names")

    if mode == 'percentage':
        fig.update_yaxes(title_text="Percentage", range=[0, 8])
    elif mode == 'count':
        fig.update_yaxes(title_text="Number of samples", range=[0, 100000])

    fig.update_layout(title_text="AudioSet labels partition ({} mode)".format(mode),
                       title_font_size=30)
    fig.write_image("graphs/audioset_{}_labels_{}_partition_mode_{}_bis.png".format(data_type, num_classes, mode), width=2000, height=1000)
    fig.show()


def plot_categorical_labels_proportion_partition(data_path, labels=labels, data_type='eval', mode='count'):
    num_classes = len(labels)
    if data_type == 'train':
        file_paths = list(
            Path(data_path / PosixPath('balanced_train') / PosixPath('txt_{}'.format(num_classes))).rglob(f"*.txt"))
        file_paths.extend(list(
            Path(data_path / PosixPath('unbalanced_train') / PosixPath('txt_{}'.format(num_classes))).rglob(f"*.txt")))
    else:
        file_paths = list(
            Path(data_path / PosixPath(data_type) / PosixPath('txt_{}'.format(num_classes))).rglob(f"*.txt"))
    print(file_paths)
    print(len(file_paths))
    all_labels = np.zeros(num_classes)
    for txt_path in tqdm(file_paths, total=len(file_paths)):
        txt_labels = parse_transcript(txt_path)
        all_labels += np.array(txt_labels)
    print(all_labels)

    return 0



def build_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_audioset_folder', type=str, default='/home/coml/Documents/Victoria/noise_classifier/deepspeech_model2/data/audioset')
    parser.add_argument('--data_type', type=str,
                        default='train', choices=['eval', 'unbalanced_train', 'balanced_train', 'train'])
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = build_argparse()
    # plot_all_labels_proportion_partition(args.path_to_audioset_folder, data_type=args.data_type, mode='percentage')
    plot_all_labels_proportion_partition(args.path_to_audioset_folder, data_type=args.data_type, mode='count')