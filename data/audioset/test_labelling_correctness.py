import unittest
import numpy as np
import pandas as pd
import os
import json
import random


class TestLabellingCorrectNess(unittest.TestCase):

    def __init__(self, data_type='eval', path_to_audioset='../audioset', path_to_labels='../../labels_183.json'):
        self.data_path = path_to_audioset

        with open(path_to_labels, 'r') as labels_file:
            self.labels = json.load(labels_file)
        self.num_classes = len(self.labels)

        with open(os.path.join(path_to_audioset, 'audioset_{}_manifest_{}.json'.format(data_type, self.num_classes)), 'r') as json_file:
            self.meta_info = json.load(json_file)['samples']
        self.csv_meta_info = pd.read_csv(os.path.join(path_to_audioset, 'csv'))


    def parse_transcript(self, transcript_path):
        with open(transcript_path, 'r', encoding='utf8') as transcript_file:
            # transcript = transcript_file.read().replace('\n', '')
            transcript = transcript_file.read().replace('\n', '')
        # transcript = list(filter(None, [self.labels_map.get(x) for x in list(transcript)]))
        new_transcript = []
        for x in transcript:
            if x not in ['', ' ', ',', ']', '[', '"']:
                new_transcript.append(int(x))
        return new_transcript

    def test_labelling(self):

        for i in range(10):
            index = random.randint(0, len(self.meta_info))
            transcript_path = self.meta_info[index]['transcript_path']
            video_id = transcript_path.split('/')[-1][:-4]
            transcripted_labels = np.array(self.parse_transcript(transcript_path))
            labels_index = np.where(transcripted_labels==1)
            corresponding_labels = [self.labels[i] for i in labels_index]

            csv_video_id_labels = self.csv_meta_info[self.csv_meta_info['YTID'] == video_id]['translated_positive_labels']
            csv_video_id_labels = csv_video_id_labels.split(',')

            print('Video {}: encoding labels {} csv_labels {}'.format(video_id, corresponding_labels, csv_video_id_labels))
            self.assertEqual(csv_video_id_labels, corresponding_labels)


if __name__ == '__main__':
    unittest.main()