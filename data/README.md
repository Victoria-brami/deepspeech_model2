# Data Preprocessing

1. The first thing the user has to do is to build the CSV file that will contain all the desired labels.
   From the example given in `class_labels_indices_183.csv`, the user rebuilts another files with the number of selected labels as a suffix.
   
2. We train deepspeech here on a combination of **Audioset** and **freesound dataset**. To generate sound and annotations files,
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


3. In a third step, the user creates the labels for freesound files as well by running the command:

    ```
    python data/preprocess_freesound_files.py 
        --
    ```
4. Finally, Audioset and freesound manifest can be merged thanks to `build_train_set.py` to create a larger
   training and validation set.
   

5. As some files in Audioset are corrupted, they need to be removed from the manifests. The user is invited to run the command:
    ```
    python data/filter_non_existing_sound_files.py 
        --
    ```
   in order to find out those corrupted files and remove them.
