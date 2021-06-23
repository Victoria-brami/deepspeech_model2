import json
import argparse
import os
import random


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
