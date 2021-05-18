# import required modules
from os import path
from pydub import AudioSegment
import os

# assign files
input_file = "/home/coml/Téléchargements/triplet_t00000103764.mp3"
output_file = "/home/coml/Téléchargements/triplet_00000103764.wav"
OUTPUT_PATH = "/home/coml/Téléchargements/wavs_triplets"

# convert mp3 file to wav file
sound = AudioSegment.from_mp3(input_file)
sound.export(output_file, format="wav")

def convert_to_wav(data_path, input_file, output_path):

    os.makedirs(output_path, exist_ok=True)

    name = input_file.split('_')[1][1:]
    real_name = name.split('.')[0]
    output_file = 'triplet_' + real_name + '.wav'

    input_filename = os.path.join(data_path, input_file)
    output_filename = os.path.join(output_path, output_file)

    sound = AudioSegment.from_mp3(input_filename)
    sound.export(output_filename, format="wav")


for (data_path, _, files) in os.walk('/home/coml/Téléchargements/TIMIT_stimuli'):
    print(data_path)
    for file in files:
        if file.startswith('triplet') and file.endswith('mp3'):
            convert_to_wav(data_path, file, output_path=OUTPUT_PATH)
        # print(file)