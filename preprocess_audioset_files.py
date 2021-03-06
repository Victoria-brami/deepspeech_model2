import os
import requests
from data.audioset.correct_csv_files_format import *
from pathlib import Path, PosixPath
from data.audioset.download_audioset_sound_files import download_audioset_sound_files
from data.audioset.create_audioset_labels import _parse_labels, create_manifest, check_manifest_length

PYTHONPATH = os.getcwd()


def build_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_audioset_folder', '-f', default='/gpfsscratch/rech/rnt/uuj49ar/audioset')
    parser.add_argument('--download_from_jeanzay', '-jz', default=True)
    parser.add_argument('--download_from_web', '-wb', default=False)
    parser.add_argument('--num_orig_classes', '-nbo', default=527)
    parser.add_argument('--num_classes', '-nb', default=183)
    parser.add_argument('--data_types', nargs='+', default=['balanced_train', 'eval', 'unbalanced_train'])

    args = parser.parse_args()
    return args



def download_labels(path_to_audioset_folder):
    output_dir = Path(path_to_audioset_folder) / PosixPath('csv')
    labels_url = requests.get(
        'http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv')
    labels_url_content = labels_url.content
    labels_filename = 'class_labels_indices_527.csv'
    labels_csv_file = open(os.path.join(output_dir, labels_filename), 'wb')
    labels_csv_file.write(labels_url_content)


def download_labels_from_jeanzay(path_to_jeanzay_audioset_folder, path_to_audioset_project_folder):

    if not os.path.exists(path_to_audioset_project_folder):
        os.mkdir(path_to_audioset_project_folder)
    os.makedirs(path_to_audioset_project_folder, exist_ok=True)
    os.makedirs(os.path.join(path_to_audioset_project_folder, 'csv'), exist_ok=True)

    current_filename = 'class_labels_indices.csv'
    new_filename = 'class_labels_indices_527.csv'
    current_path = os.path.join(path_to_jeanzay_audioset_folder, current_filename)
    destination_path = os.path.join(path_to_audioset_project_folder, 'csv', new_filename)

    os.system('cp {} {}'.format(current_path, os.path.join(path_to_audioset_project_folder, 'csv')))
    os.system('mv {} {}'.format(os.path.join(path_to_audioset_project_folder, 'csv', current_filename), destination_path))


def download_annotations_files(path_to_audioset_folder, num_classes=None):
    annotations_output_dir = Path(path_to_audioset_folder) / PosixPath('csv')
    os.makedirs(annotations_output_dir, exist_ok=True)

    balanced_train_annotations_url = requests.get(
        'http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/balanced_train_segments.csv')
    unbalanced_train_annotations_url = requests.get(
        'http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/unbalanced_train_segments.csv')
    eval_annotations_url = requests.get(
        'http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/eval_segments.csv')

    # Get filenames
    if num_classes is None:
        num_classes = 527

    balanced_train_filename = 'balanced_train_segments_{}.csv'.format(num_classes)
    unbalanced_train_filename = 'unbalanced_train_segments_{}.csv'.format(num_classes)
    eval_filename = 'eval_segments_{}.csv'.format(num_classes)

    balanced_train_annotations_url_content = balanced_train_annotations_url.content
    unbalanced_train_annotations_url_content = unbalanced_train_annotations_url.content
    eval_annotations_url_content = eval_annotations_url.content

    balanced_train_csv_file = open(os.path.join(annotations_output_dir, balanced_train_filename), 'wb')
    unbalanced_train_csv_file = open(os.path.join(annotations_output_dir, unbalanced_train_filename), 'wb')
    eval_csv_file = open(os.path.join(annotations_output_dir, eval_filename), 'wb')

    balanced_train_csv_file.write(balanced_train_annotations_url_content)
    unbalanced_train_csv_file.write(unbalanced_train_annotations_url_content)
    eval_csv_file.write(eval_annotations_url_content)


def download_annotations_files_from_jeanzay(path_to_jeanzay_audioset_folder,
                                            path_to_audioset_project_folder,
                                            data_type):

    current_filename = '{}_segments.csv'.format(data_type)
    new_filename = '{}_segments_527.csv'.format(data_type)
    current_path = os.path.join(path_to_jeanzay_audioset_folder, current_filename)
    destination_path = os.path.join(path_to_audioset_project_folder, 'csv', new_filename)

    # Copy annotations to project path
    os.system('cp {} {}'.format(current_path, os.path.join(path_to_audioset_project_folder, 'csv')))
    os.system('mv {} {}'.format(os.path.join(path_to_audioset_project_folder, 'csv', current_filename), destination_path))


def preprocess_audioset(args):

    # 1)ter) Move filtered labels to right directory
    labels_current_path = os.path.join(PYTHONPATH, 'class_labels_indices_{}.csv'.format(args.num_classes))
    labels_destination_path = os.path.join(args.path_to_audioset_folder, 'csv')
    os.system('cp {} {}'.format(labels_current_path, labels_destination_path))
    print(' 1)  Filtered Labels moved to the right directory ! ')
    print('cp {} {}'.format(labels_current_path, labels_destination_path))

    if args.download_from_web:
        # 1) Download audioset csv files
        download_annotations_files(args.path_to_audioset_folder)
        print()
        print(' 1)  All annotations downloaded ! ')
        # WORKS

        # 1)bis) Download labels file
        download_labels(args.path_to_audioset_folder)
        print(' 2)  Labels downloaded ! ')

    index = 1

    if args.download_from_jeanzay:
        print()
        download_labels_from_jeanzay('/gpfsdswork/dataset/AudioSet', args.path_to_audioset_folder)
        correct_labels_format(args.path_to_audioset_folder)

    for data_type in args.data_types:  # 'balanced_train', 'unbalanced_train', :

        if args.download_from_jeanzay:  # Audioset paths stored in DSDIR on JEANZAY
            index += 1
            download_annotations_files_from_jeanzay('/gpfsdswork/dataset/AudioSet', args.path_to_audioset_folder,
                                                    data_type)
            print(' {})  {}: CSV  annotation Files downloaded ! '.format(index, data_type))


        # 2) Re-format csv files
        convert_false_csv_files_to_dataframes(args.path_to_audioset_folder, data_type)
        print(' {})  {}: CSV Files rightly converted ! '.format(index + 1, data_type))
        align_translated_labels_with_csv_files(args.path_to_audioset_folder, data_type, args.num_orig_classes)
        print('\n {})  {}: CSV Files rightly aligned ! '.format(index + 2, data_type))

        # 3) Filter interesting labels
        filter_dataset_on_non_human_labels(args.path_to_audioset_folder, data_type, num_classes=args.num_classes)
        print(' {})  {}: CSV Files rightly filtered ! '.format(index + 3, data_type))

        # 5) Create labels
        _parse_labels(args.path_to_audioset_folder, data_type=data_type, num_classes=args.num_classes)
        print(' {})  {}: All labels created ! '.format(index + 4, data_type))

        # 6) create manifests
        create_manifest(args.path_to_audioset_folder, data_type=data_type, size='small', num_classes=args.num_classes)
        print(' {})  {}: Small Json manifests created ! '.format(index + 5, data_type))

        # 6) create manifests
        create_manifest(args.path_to_audioset_folder, data_type=data_type, num_classes=args.num_classes)
        print(' {})  {}: Json manifests created ! '.format(index + 6, data_type))

        # 7) Check Manifest length
        check_manifest_length(args.path_to_audioset_folder, size=None, data_type=data_type, num_classes=args.num_classes)
        check_manifest_length(args.path_to_audioset_folder, size='small', data_type=data_type, num_classes=args.num_classes)
        print()

        index += 7


# 7) Merge manifests (TO DO !!!)

if __name__ == '__main__':
    args = build_arguments()
    args.download_from_web = False
    args.download_from_jeanzay = True
    # args.path_to_audioset_folder = '/home/coml/Documents/Victoria/noise_classifier/deepspeech_model2/data/audioset'
    preprocess_audioset(args)
