# MM-USElecDeb60to16 

## Folder description: multimodal-dataset

- [files](files): the folder contains all the files necessary to reconstruct the audio clips of the multimodal dataset.
  - [alignment_results](files/alignment_results): The folder contains the results of aligning audio files with 
    transcripts. There are several subfolders (one per debate) and each subfolder contains as many _.json_ files as the
    number of chunks into which the original audio file of the debate was divided. The alignment was performed with 
    _aeneas_. For more details on the structure of _.json_ files, please visit the 
    [aenas documentation](https://www.readbeyond.it/aeneas/docs/).
  - [datasets](files/datasets): the folder contains several sub-folders (one per debate). Each sub-folder contains 
    several _.csv_ files representing the intermediate results of the final dataset construction process. In addition, 
    for each debate there is a _duplicates.txt_ file containing the duplicated sentences and the number of occurrences.
    In addition, there is a YesWeCan folder containing the contents of the original dataset
    ([USElecDeb60to16](https://github.com/ElecDeb60To16/Dataset)). 
  - [MM-USElecDeb60to16](files/MM-USElecDeb60to16): is the official dataset folder. It currently contains the _.csv_ 
    file corresponding to the new dataset and a `audio_clips` folder that will be created/populated after downloading 
    and processing the audio files with the files in the `files/audio_clips` folder. 
  - [transcripts](files/transcripts): the folder contains several sub-folders (one per debate). Each sub-folder contains: 
    - the original transcript
    - the plain version of the transcript 
    - a `splits` sub-folder containing the portions of text corresponding to each chunk
  - [debug.csv](files/debug.csv): is a debugging file containing the necessary information for downloading and trimming 
    the audio files of two debates. The columns in this file are the stars of the `dictionary.csv` file
  - [dicionary.csv](files/dictionary.csv): this file contains the information needed to download and trim all debates. 
    The columns in the dataset are: 
    - `id`: debate identifier number. Corresponds to the `id` of the debates in `USEleDeb60to16` and `MM-USElecDeb60to16`
    - `link`: link to the corresponding YouTube video of the debate 
    - `startMin`: number of minutes to be cut from the beginning of the file 
    - `startSec`: number of seconds to be cut from the beginning of the file 
    - `endMin`: number of minutes to be cut from the end of the file 
    - `endSec`: number of seconds to be cut from the end of the file 
  - [run_aeneas](files/run_aeneas): folder containing the bash script needed to run _aeneas_
  - [audio_pipepline.py](files/audio_pipeline.py): Python script to perform operations for recontructing the audio 
    part of MM-USElecDeb60to16 
  - [full_pipeline.py](files/full_pipeline.py): Python script to perform all dataset construction operations 
  - (i.e. text part, audio part, creation of folders, datasets and alignment) 
  - [utils.py](files/utils.py): contains all the functions needed to construct the dataset.

## Usage
- Download the folder `multimodal-debates`
- Install all the required packages. List of required packages can be found at `requirements.txt`(files/requirements.txt)
- Run [audio_pipepline.py](files/audio_pipeline.py). While running the script, several folders will be created: 
  - [audio_clips](files/audio_clips): After the clips have been generated, this folder will contain
    several sub-folders (one per debate), each of which will contain as many clips as there are text samples in the 
    dataset for the specific debate.
  - [debates_audio_recordings](files/debates_audio_recordings): folder is empty and will be populated with several 
    sub-folders (one per debate). Each subfolder will contain:
    - a _splits_ subfolder containing the new audio files after 
      splitting into chunks
    - a version of the audio file __trim.wav_ corresponding to the trimmed version of the original 
    - the original audio file
    
- The dataset will be available in the following folder [MM-USElecDeb60to16](files/MM-USElecDeb60to16). 

## Dataset Description
In addition to the information present in the original dataset (please, see 
[USElecDeb60to16](https://github.com/ElecDeb60To16/Dataset) for detailed information), MM-USElecDeb60to16 contains 3 
additional columns: 
- `NewBegin`: the number of seconds corresponding to the beginning of the phrase with respect to the duration 
  of the trimmed original audio file 
- `NewEnd`: the number of seconds corresponding to the end of the phrase with respect to the duration of the 
   trimmed original audio file 
- `idClip`: the identifier of the audio clip corresponding to the sentence. This `id` is needed to reconstruct the 
  audio-text pairs for each part of the speech
