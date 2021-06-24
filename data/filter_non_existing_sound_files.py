import json
import argparse
import os
import random
import torchaudio
import sys

def display_non_existing_files(path_to_json_file, display_filename):

    with open(path_to_json_file, 'r') as json_file:
        dataset = json.load(json_file)['samples']

    number_of_missing_files = 0
    missing_files = []
    non_missing_files = []

    for sample in dataset:
        wav_file = sample['wav_path']

        if not os.path.isfile(wav_file):
            missing_files.append(wav_file)
            number_of_missing_files += 1
        else:
            non_missing_files.append(wav_file)

    randomlist = random.sample(range(0, number_of_missing_files), 10)
    non_randomlist = random.sample(range(0, len(non_missing_files)), 10)
    print('    Number of Missing files: {} / {}  \n'.format(number_of_missing_files, len(dataset)))

    if display_filename:
        for i in range(10):
            index = randomlist[i]
            print('\n    {}: {}'.format(index, missing_files[index]))
        print(' Non missing files: ')
        for i in range(10):
            index = non_randomlist[i]
            print('      {}: {}'.format(index, non_missing_files[index]))



def parse_transcript(transcript_path):
    with open(transcript_path, 'r', encoding='utf8') as transcript_file:
        transcript = transcript_file.read().replace('\n', '')
    new_transcript = []
    for x in transcript:
        if x not in ['', ' ', ',', ']', '[', '"']:
            new_transcript.append(int(x))
    return new_transcript


def load_audio(path):
    sound, sample_rate = torchaudio.load(path)
    if sound.shape[0] == 1:
        sound = sound.squeeze()
    else:
        sound = sound.mean(axis=0)  # multiple channels, average
    return sound.numpy()



def check_all_labels_contents(path_to_json_file, num_classes=183, start_index=17420):

    with open(path_to_json_file, 'r') as json_file:
        dataset = json.load(json_file)['samples']

    sample_index = 0

    for sample in dataset[start_index:]:
        wav_file = sample['wav_path']
        txt_file = sample['transcript_path']

        label = parse_transcript(txt_file)
        wav = load_audio(wav_file)

        sys.stdout.write('\r  Processing on wav file {} / {}'.format(start_index + sample_index, len(dataset[start_index:])))

        if len(label) != num_classes:
            print('     Problem with the labels in the file:  ', txt_file)
        if wav.shape[0] < 100000:
            print('       Audio size is: ', wav.shape, wav_file, start_index + sample_index)
        sample_index += 1


def remove_corrupted_files(path_to_json_file, corrupted_list, num_classes=183):

    with open(path_to_json_file, 'r') as json_file:
        dataset = json.load(json_file)

    sample_index = 0

    new_dataset = []

    for sample in dataset['samples']:
        sys.stdout.write('\r  Processing on wav file {} / {}'.format(sample_index, len(dataset)))

        wav_name = sample['wav_path']

        if wav_name not in corrupted_list:
            new_dataset.append(wav_name)

    dataset['samples'] = new_dataset

    with open(path_to_json_file, 'w') as new_json_file:
        json.dump(dataset, new_json_file, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--display_missing_files', '-d')
    args = parser.parse_args()
    print(args.display_missing_files)
    print('\n  1) Training Set: ')
    display_non_existing_files('/gpfsscratch/rech/rnt/uuj49ar/train_manifest_183.json', args.display_missing_files)
    print('\n  2) Validation Set: ')
    display_non_existing_files('/gpfsscratch/rech/rnt/uuj49ar/validation_manifest_183.json', args.display_missing_files)
    print('\n  3) Test Set: ')
    display_non_existing_files('/gpfsscratch/rech/rnt/uuj49ar/test_manifest_183.json', args.display_missing_files)

    print('\n  4) SMALL Training Set: ')
    check_all_labels_contents('/gpfsscratch/rech/rnt/uuj49ar/small_train_no_noise_manifest_183.json', num_classes=183)
    print('\n  5) SMALL Validation Set: ')
    check_all_labels_contents('/gpfsscratch/rech/rnt/uuj49ar/small_validation_no_noise_manifest_183.json', num_classes=183)
    print('\n  6) SMALL Test Set: ')
    check_all_labels_contents('/gpfsscratch/rech/rnt/uuj49ar/small_test_no_noise_manifest_183.json', num_classes=183)


    CORRUPTED_LIST = ['/gpfsdswork/dataset/AudioSet/unbalanced_train/-/-zVGs8QmK1I_210.000_220.000.wav']
