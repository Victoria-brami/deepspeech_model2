import json
import os
import numpy as np
import plotly.graph_objects as go

PATH_TO_TRAIN_DATASET = '/home/coml/Documents/Victoria/noise_classifier/deepspeech_model_old/data/Freesound_dataset/for_oberon'
PATH_TO_TRAIN_DATASET = '/scratch2/vbrami/noise_classifier/deepspeech_model2/data/Freesound_dataset'


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

if __name__ == '__main__':
    with open(os.path.join(PATH_TO_TRAIN_DATASET, 'freesound_train_manifest_filtered.json'), 'r') as json_file:
        train_data = json.load(json_file)

    print('NUMBER OF SAMPLES: ', len(train_data['samples']))

    labels_partition = np.zeros(59)
    samples_per_labels = [[]*59]

    # Extract transcript file
    for sample in train_data['samples']:

        transcript = parse_transcript(sample['transcript_path'])
        labels_partition += np.array(transcript)
        indexes = np.where(np.array(transcript) == 1)[0]

        for idx in indexes:
            samples_per_labels[idx].append(sample['transcript_path'].split('/')[-1])

    # Create Figure
    fig = go.Figure(data=go.Bar(x=[i for i in range(1, 60)], y=labels_partition))
    fig.layout.title.text = "Dataset Labels partition"
    fig.update_xaxes(title_text="Labels index")
    fig.update_yaxes(title_text="Number of samples")

    fig.write_image(
        '/scratch2/vbrami/noise_classifier/deepspeech_model2/data/Freesound_dataset/labels_partition.png',
        width=1000, height=800)