import pandas as pd
import argparse
from tqdm.notebook import tqdm
import sys
import os

""" 
    From the labels CSV File, add a new column ato audioset sounds csv  file with labels transcription
"""


def preprocess_csv_files_3(path_to_csv):
    with open(path_to_csv, 'r', encoding='utf8') as transcript_file_2:
        transcript2 = transcript_file_2.read().replace(" ", "")
    f2 = open(path_to_csv, 'w')
    f2.write(transcript2)


def translate_labels(csv_data_file, labels_encoding, path_to_csv_file):
    new_translated_column = []

    print('\n Before Translation \n', csv_data_file.head())

    for (i, row) in tqdm(csv_data_file.iterrows(), total=csv_data_file.shape[0]):
        print(row)
        list_of_labels = row['positive_labels'].split(',')
        traduced_list_of_labels = []
        for label in list_of_labels:
            traduced_list_of_labels.append(labels_encoding[labels_encoding['mid'] == label, ['display_name']])
        traduced_labels = ','.join(traduced_list_of_labels)
        new_translated_column.append(traduced_labels)

    csv_data_file['translated_positive_labels'] = new_translated_column
    csv_data_file.to_csv(path_to_csv_file, index=False)

    print('\n After Translation \n', csv_data_file.head())

    return None


def convert_false_csv_files_to_dataframes(path_to_folder, data_type, num_classes=527):
    new_data = dict(YTID=[], start_seconds=[], end_seconds=[], positive_labels=[])

    path_to_csv = os.path.join(path_to_folder, 'csv', '{}_segments_{}.csv').format(data_type, num_classes)

    file1 = open(path_to_csv, 'r')
    all_lines = file1.readlines()
    for line in all_lines[3:]:
        list_of_items = line.split(' ')
        new_data['YTID'].append(str(list_of_items[0][:-1]))
        new_data['start_seconds'].append(float(list_of_items[1][:-1]))
        new_data['end_seconds'].append(float(list_of_items[2][:-1]))
        new_data['positive_labels'].append(str(list_of_items[3][1:-2]))

    new_data = pd.DataFrame(new_data)
    new_data.to_csv(path_to_csv, index=False)

    return None


def correct_labels_format(path_to_audioset_folder):
    # Ensure labels file in the right format
    path_to_labels = os.path.join(path_to_audioset_folder, 'csv', 'class_labels_indices_527.csv')
    labels = pd.read_csv(path_to_labels)
    for i in range(len(labels)):
        lab = labels['display_name'][i]
        new_lab = lab.replace(', ', '_and_')
        new_new_lab = new_lab.replace(' ', '_')
        labels['display_name'][i] = new_new_lab
    labels.to_csv(path_to_labels, index=False)


def align_translated_labels_with_csv_files(path_to_audioset_folder, data_type, num_classes=527):
    path_to_csv_files = os.path.join(path_to_audioset_folder, 'csv',
                                     '{}_segments_{}.csv'.format(data_type, num_classes))
    path_to_labels = os.path.join(path_to_audioset_folder, 'csv', 'class_labels_indices_{}.csv'.format(num_classes))
    data = pd.read_csv(path_to_csv_files)
    labels = pd.read_csv(path_to_labels)
    translated_list_of_labels = []

    for i in range(len(data)):
        sys.stdout.write('\r  treating all labels .... [{} / {}]'.format(i, len(data)))
        data_labels = data['positive_labels'][i].split(',')
        translated_lab_list = []
        for lab in data_labels:
            translated_lab = labels[labels['mid'] == lab]
            translated_lab = translated_lab['display_name'].item()
            translated_lab_list.append(str(translated_lab))
        translated_labels = ','.join(translated_lab_list)
        translated_list_of_labels.append(translated_labels)
    data['translated_positive_labels'] = translated_list_of_labels
    data.to_csv(path_to_csv_files, index=False)


def filter_dataset_on_non_human_labels(path_to_audioset_folder, data_type, num_classes):
    path_to_csv_file = os.path.join(path_to_audioset_folder, 'csv', '{}_segments_527.csv').format(data_type)
    data = pd.read_csv(path_to_csv_file)

    path_to_filtered_labels = os.path.join(path_to_audioset_folder, 'csv',
                                           'class_labels_indices_{}.csv'.format(num_classes))
    labels = pd.read_csv(path_to_filtered_labels)
    list_of_filtered_labels = list(labels['display_name'])

    filtered_path = path_to_csv_file.replace('_527.csv', '_{}.csv'.format(len(list_of_filtered_labels)))

    new_data = dict(YTID=[], start_seconds=[], end_seconds=[],
                    positive_labels=[], translated_positive_labels=[])

    for i in range(len(data)):
        number_of_found_labels = 0
        list_of_labels = data['translated_positive_labels'][i].split(',')
        # print(list_of_labels)
        for lab in list_of_labels:
            if lab in list_of_filtered_labels:
                number_of_found_labels += 1
        if number_of_found_labels == len(list_of_labels):
            new_data['YTID'].append(data['YTID'][i])
            new_data['start_seconds'].append(data['start_seconds'][i])
            new_data['end_seconds'].append(data['end_seconds'][i])
            new_data['positive_labels'].append(data['positive_labels'][i])
            new_data['translated_positive_labels'].append(data['translated_positive_labels'][i])

    new_data = pd.DataFrame(new_data)
    new_data.to_csv(filtered_path, index=False)


def build_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_labels',
                        default='/home/coml/Documents/Victoria/noise_classifier/deepspeech_model2/data/audioset/csv/class_labels_indices.csv')
    parser.add_argument('--path_to_filtered_labels',
                        default='/home/coml/Documents/Victoria/noise_classifier/deepspeech_model2/data/audioset/csv/filtered_class_labels_indices.csv')
    parser.add_argument('--path_to_csv_file',
                        default='/home/coml/Documents/Victoria/noise_classifier/deepspeech_model2/data/audioset/csv/unbalanced_train_segments.csv')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = build_argparse()

    # convert_false_csv_files_to_dataframes(args.path_to_csv_file)
    # align_translated_labels_with_csv_files(args.path_to_labels, args.path_to_csv_file)
    filter_dataset_on_non_human_labels(args.path_to_csv_file, args.path_to_filtered_labels)

    """
    convert_false_csv_files_to_dataframes(args.path_to_csv_file)
    csv_data_file = pd.read_csv(args.path_to_csv_file)
    labels_encoding =
    translate_labels(csv_data_file, labels_encoding, path_to_csv_file)
    print(csv_data_file.head())"""
