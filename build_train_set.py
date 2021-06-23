import json
import argparse
import os


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

def merge_audioset_manifests(path_to_data_folder, data_type, num_classes):

    manifest = {
        'root_path': path_to_data_folder,
        'samples': []
    }

    if data_type == 'train' or data_type == 'validation':
        audioset_balanced_train_file = os.path.join(path_to_data_folder, 'audioset',
                                                    'audioset_balanced_train_manifest_{}.json'.format(num_classes))
        audioset_unbalanced_train_file = os.path.join(path_to_data_folder, 'audioset',
                                                      'audioset_unbalanced_train_manifest_{}.json'.format(num_classes))
        freesound_train_curated_file = os.path.join(path_to_data_folder, 'freesound',
                                                    'freesound_train_curated_manifest_{}.json'.format(num_classes))
        freesound_train_noisy_file = os.path.join(path_to_data_folder, 'freesound',
                                                  'freesound_train_noisy_manifest_{}.json'.format(num_classes))

        with open(audioset_balanced_train_file, 'r') as json_file1:
            audioset_b_train = json.load(json_file1)['samples']
        with open(audioset_unbalanced_train_file, 'r') as json_file2:
            audioset_ub_train = json.load(json_file2)['samples']
        with open(freesound_train_curated_file, 'r') as json_file3:
            freesound_c_train = json.load(json_file3)['samples']
        with open(freesound_train_noisy_file, 'r') as json_file4:
            freesound_n_train = json.load(json_file4)['samples']

        if data_type == 'train':
            for sample_idx_1 in range(int(len(audioset_b_train) * 0.9)):
                manifest['samples'].append(audioset_b_train[sample_idx_1])
            for sample_idx_2 in range(int(len(audioset_ub_train) * 0.9)):
                manifest['samples'].append(audioset_ub_train[sample_idx_2])
            for sample_idx_3 in range(int(len(freesound_c_train) * 0.9)):
                manifest['samples'].append(freesound_c_train[sample_idx_3])
            for sample_idx_4 in range(int(len(freesound_n_train) * 0.9)):
                manifest['samples'].append(freesound_n_train[sample_idx_4])

        else: # validation
            for sample_idx_1 in range(int(len(audioset_b_train) * 0.9), len(audioset_b_train)):
                manifest['samples'].append(audioset_b_train[sample_idx_1])
            for sample_idx_2 in range(int(len(audioset_ub_train) * 0.9), len(audioset_ub_train)):
                manifest['samples'].append(audioset_ub_train[sample_idx_2])
            for sample_idx_3 in range(int(len(freesound_c_train) * 0.9), len(freesound_c_train)):
                manifest['samples'].append(freesound_c_train[sample_idx_3])
            for sample_idx_4 in range(int(len(freesound_n_train) * 0.9), len(freesound_n_train)):
                manifest['samples'].append(freesound_n_train[sample_idx_4])


    elif data_type == 'test':
        audioset_eval_file = os.path.join(path_to_data_folder, 'audioset',
                                                      'audioset_eval_manifest_{}.json'.format(num_classes))
        freesound_test_file = os.path.join(path_to_data_folder, 'freesound',
                                                    'freesound_test_manifest_{}.json'.format(num_classes))
        with open(audioset_eval_file, 'r') as json_file2:
            audioset_eval = json.load(json_file2)['samples']
        with open(freesound_test_file, 'r') as json_file3:
            freesound_test = json.load(json_file3)['samples']

        for sample_idx_2 in range(len(audioset_eval)):
            manifest['samples'].append(audioset_eval[sample_idx_2])
        for sample_idx_3 in range(len(freesound_test)):
            manifest['samples'].append(freesound_test[sample_idx_3])

    elif data_type == 'small_train' or data_type == 'small_validation':
        audioset_balanced_train_file = os.path.join(path_to_data_folder, 'audioset',
                                                    'audioset_small_balanced_train_manifest_{}.json'.format(num_classes))
        audioset_unbalanced_train_file = os.path.join(path_to_data_folder, 'audioset',
                                                      'audioset_small_unbalanced_train_manifest_{}.json'.format(num_classes))
        freesound_train_curated_file = os.path.join(path_to_data_folder, 'freesound',
                                                    'freesound_train_curated_manifest_{}.json'.format(num_classes))
        freesound_train_noisy_file = os.path.join(path_to_data_folder, 'freesound',
                                                  'freesound_train_noisy_manifest_{}.json'.format(num_classes))

        with open(audioset_balanced_train_file, 'r') as json_file1:
            audioset_b_train = json.load(json_file1)['samples']
        with open(audioset_unbalanced_train_file, 'r') as json_file2:
            audioset_ub_train = json.load(json_file2)['samples']
        with open(freesound_train_curated_file, 'r') as json_file3:
            freesound_c_train = json.load(json_file3)['samples']
        with open(freesound_train_noisy_file, 'r') as json_file4:
            freesound_n_train = json.load(json_file4)['samples']

        if data_type == 'small_train':
            for sample_idx_1 in range(int(len(audioset_b_train) * 0.9)):
                manifest['samples'].append(audioset_b_train[sample_idx_1])
            for sample_idx_2 in range(int(len(audioset_ub_train) * 0.9)):
                manifest['samples'].append(audioset_ub_train[sample_idx_2])
            for sample_idx_3 in range(int(len(freesound_c_train) * 0.9)):
                manifest['samples'].append(freesound_c_train[sample_idx_3])
            for sample_idx_4 in range(int(len(freesound_n_train) * 0.9)):
                manifest['samples'].append(freesound_n_train[sample_idx_4])

        else: # validation
            for sample_idx_1 in range(int(len(audioset_b_train) * 0.9), len(audioset_b_train)):
                manifest['samples'].append(audioset_b_train[sample_idx_1])
            for sample_idx_2 in range(int(len(audioset_ub_train) * 0.9), len(audioset_ub_train)):
                manifest['samples'].append(audioset_ub_train[sample_idx_2])
            for sample_idx_3 in range(int(len(freesound_c_train) * 0.9), len(freesound_c_train)):
                manifest['samples'].append(freesound_c_train[sample_idx_3])
            for sample_idx_4 in range(int(len(freesound_n_train) * 0.9), len(freesound_n_train)):
                manifest['samples'].append(freesound_n_train[sample_idx_4])


    elif data_type == 'small_test':
        audioset_eval_file = os.path.join(path_to_data_folder, 'audioset',
                                                      'audioset_small_eval_manifest_{}.json'.format(num_classes))
        freesound_test_file = os.path.join(path_to_data_folder, 'freesound',
                                                    'freesound_test_manifest_{}.json'.format(num_classes))
        with open(audioset_eval_file, 'r') as json_file2:
            audioset_eval = json.load(json_file2)['samples']
        with open(freesound_test_file, 'r') as json_file3:
            freesound_test = json.load(json_file3)['samples']

        for sample_idx_2 in range(len(audioset_eval)):
            manifest['samples'].append(audioset_eval[sample_idx_2])
        for sample_idx_3 in range(len(freesound_test)):
            manifest['samples'].append(freesound_test[sample_idx_3])

    output_path = os.path.join(path_to_data_folder, '{}_manifest_{}.json'.format(data_type, num_classes))

    with open(output_path, 'w') as json_manifest_file:
        json.dump(manifest, json_manifest_file, indent=4)

    # output_path.write_text(json.dumps(manifest, indent=4), encoding='utf8')

def dataset_builder(args):
    merge_audioset_manifests(args.path_to_data_folder, 'train', args.num_classes)
    merge_audioset_manifests(args.path_to_data_folder, 'validation', args.num_classes)
    merge_audioset_manifests(args.path_to_data_folder, 'test', args.num_classes)
    merge_audioset_manifests(args.path_to_data_folder, 'small_train', args.num_classes)
    merge_audioset_manifests(args.path_to_data_folder, 'small_validation', args.num_classes)
    merge_audioset_manifests(args.path_to_data_folder, 'small_test', args.num_classes)

def build_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', '-nb', default=183)
    parser.add_argument('--path_to_data_folder', '-d', default=None)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = build_arguments()
    dataset_builder(args)
