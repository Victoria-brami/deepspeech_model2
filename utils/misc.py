import numpy as np
import torch
import os
import json

def nan_and_inf_values_seeker(input, mode, threshold=1000000):
    """

    :param input: (
    :param mode: (str) either input or output
    :param threshold: (int) threshold for strange values
    :return:
    """

    mins, maxs = torch.min(input).item(), torch.max(input).item()
    isnan_identifier = torch.isnan(input)
    isinf_identifier = torch.isinf(input)

    assertion1 = abs(mins) > threshold
    assertion2 = abs(maxs) > threshold
    assertion3 = True in isnan_identifier
    assertion4 =  True in isinf_identifier

    if True in [assertion1, assertion2, assertion3, assertion4]:
        print('   Nan or Inf Values found in {}'.format(mode))


def min_and_max_output_values_seeker(output):

    mins = torch.min(output, axis=-1).values.numpy()
    maxs = torch.max(output, axis=-1).values.numpy()

    print(' MINs {} \n and \n MAXs {}'.format(mins, maxs))


def negative_values_seeker(output):

    boolean_values = output > 0

    if False in boolean_values:
        print('   Negative or Null value found in output')


def plot_output(output):
    print('\n Network output', output)


def show_model_weights(model):
    # Plot the model weights to see if they are evolving
    return None


def get_audio_file_duration(wav_path):

    split_name = wav_path.split('_')
    time_begin = int(split_name[-2].split('.')[0])
    time_end = int(split_name[-1].split('.')[0])
    return time_end - time_begin



def extract_set_overall_duration(path_to_audioset, data_type, num_classes):

    overall_time = 0

    filename = '{}_manifest_{}.json'.format(data_type, num_classes)

    with open(os.path.join(path_to_audioset, filename), 'r') as json_file:
        db = json.load(json_file)['samples']

    for sample in db:
        wav_path = sample['wav_path']
        if 'freesound' not in wav_path:
            overall_time += get_audio_file_duration(wav_path)

    hours = overall_time // 3600
    mins = int(overall_time - hours * 3600) // 60
    secs = int(overall_time - hours * 3600 - mins * 60) // 60

    print('/////////////   Overall Time in {}   //////////////////'.format(filename))
    print('\n               {} hours {} minutes {} seconds  \n'.format(hours, mins, secs))
    print('//////////////////////   END   //////////////////////// \n')

if __name__ == '__main__':
    output = torch.randn((4, 8))
    extract_set_overall_duration('/home/coml/Téléchargements', 'small_train', 182)
    extract_set_overall_duration('/home/coml/Téléchargements', 'small_validation', 182)