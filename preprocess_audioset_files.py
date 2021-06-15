import os
import requests
from data.audioset.correct_csv_files_format import *
from pathlib import Path, PosixPath
from data.audioset.download_audioset_sound_files import download_audioset_sound_files
from data.audioset.create_audioset_labels import _parse_labels, create_manifest

PYTHONPATH = os.getcwd()

def build_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_audioset_folder', '-f', default= '/gpfsscratch/rech/rnt/uuj49ar/audioset')
    parser.add_argument('--labels_filename', '-lb')
    parser.add_argument('--num_orig_classes', '-nbo', default=527)
    parser.add_argument('--num_classes', '-nb', default=183)
    args = parser.parse_args()
    return args

def download_labels(path_to_audioset_folder):
    output_dir = Path(path_to_audioset_folder) / PosixPath('csv')
    labels_url = requests.get('http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv')
    labels_url_content = labels_url.content
    labels_filename = 'class_labels_indices_527.csv'
    labels_csv_file = open(os.path.join(output_dir, labels_filename), 'wb')
    labels_csv_file.write(labels_url_content)



def download_annotations_files(path_to_audioset_folder, num_classes=None):
    annotations_output_dir = Path(path_to_audioset_folder )/ PosixPath('csv')
    os.makedirs(annotations_output_dir, exist_ok=True)

    balanced_train_annotations_url = requests.get('http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/balanced_train_segments.csv')
    unbalanced_train_annotations_url = requests.get('http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/unbalanced_train_segments.csv')
    eval_annotations_url = requests.get('http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/eval_segments.csv')

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




def preprocess_audioset(args):

    # 1) Download audioset csv files
    download_annotations_files(args.path_to_audioset_folder)
    print()
    print(' 1)  All annotations downloaded ! ')
    # WORKS

    # 1)bis) Download labels file
    download_labels(args.path_to_audioset_folder)
    print(' 2)  Labels downloaded ! ')
    # 1)ter) Move filtered labels to right directory
    labels_current_path = os.path.join(PYTHONPATH, 'class_labels_indices_{}.csv'.format(args.num_classes))
    labels_destination_path = os.path.join(args.path_to_audioset_folder, 'csv', 'class_labels_indices_{}.csv'.format(args.num_classes))
    os.system('cp {} {}'.format(labels_current_path, labels_destination_path))
    print(' 3)  Filtered Labels moved to the right directory ! ')

    for data_type in ['eval'] :#'balanced_train', 'unbalanced_train', :

        # 2) Re-format csv files
        convert_false_csv_files_to_dataframes(args.path_to_audioset_folder, data_type)
        print(' 4)  {}: CSV Files rightly converted ! '.format(data_type))
        align_translated_labels_with_csv_files(args.path_to_audioset_folder, data_type, args.num_orig_classes)
        print('\n 5)  {}: CSV Files rightly aligned ! '.format(data_type))

        # 3) Filter interesting labels
        filter_dataset_on_non_human_labels(args.path_to_audioset_folder, data_type, num_classes=args.num_classes)
        print(' 6)  {}: CSV Files rightly filtered ! '.format(data_type))

        # 4) Download audioset sound files
        download_audioset_sound_files(args.path_to_audioset_folder, data_type, args.num_classes)
        print(' 7)  {}: All sound Files downloaded ! '.format(data_type))

        # 5) Create labels
        _parse_labels(args.path_to_audioset_folder, data_type=data_type, num_classes=args.num_classes)
        print(' 8)  {}: All labels created ! '.format(data_type))


        # 6) create manifests
        output_name = 'audioset_{}_manifest_{}.json'.format(data_type, args.num_classes)
        manifest_path = args.path_to_audioset_folder
        create_manifest(args.path_to_audioset_folder, output_name, manifest_path, file_extension='wav',
                        data_type=data_type, num_classes=args.num_classes)
        print(' 9)  {}: Json manifests created ! '.format(data_type))
"""    """


    # 7) Merge manifests (TO DO !!!)

if __name__ == '__main__':
    args = build_arguments()
    preprocess_audioset(args)