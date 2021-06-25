# Data Preprocessing


We train deepspeech here on a combination of **Audioset** and **freesound dataset**. To generate sound and annotations files,
the user must run the commands:

```
python preprocess_audioset_files.py 
    --path_to_audioset_folder $DATAPATH
    --download_from_jeanzay True
    --download_from_web False
    --num_orig_classes 527
    --num_classes 183
    --data_types ['balanced_train', 'unbalanced_train', 'eval']
```

where:

- _path_to_audioset_folder_ (string) is the path in which all the audioset sound and annotations files will be stored.
- _download_from_jeanzay_
- _num_orig_classes_: the initial number of labels in google Audioset (as to say 527)
- _num_classes_: the final number of labels we will use
- _data_types_ : contains the names of the audio folders

However, the user must build at first the list of the labels he will use for dataset preprocessing. It ahas to have the 
same shape as the one described in Audioset site.


```
python data/preprocess_freesound_files.py 
    --
```