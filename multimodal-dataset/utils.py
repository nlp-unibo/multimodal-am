import shutil
import pandas as pd
import os
from tqdm import tqdm
from pydub import AudioSegment
from nltk.tokenize import sent_tokenize
import yt_dlp
from collections import Counter

AUDIO_FILE_PATH = 'files/debates_audio_recordings/'
DELETED_ID = ['13_1988, 17_1992, 42_2016, 43_2016']


def youtube_download(id: list, link: list) -> None:
    """

    :param id: list of strings representing debates IDs
    :param link: list of strings representing the urls to the YouTube videos of the debates
    :return: None. The function populates the folder 'files/debates_audio_recordings' by creating a folder for each
             debate. Each folder contains the audio file extracted from the corresponding video
    """
    map_debate_link = dict(zip(id, link))
    for doc, link in tqdm(map_debate_link.items()):
        audio_path = AUDIO_FILE_PATH + doc
        if not os.path.exists('files/debates_audio_recordings'):
            os.makedirs('files/debates_audio_recordings')

        os.makedirs(audio_path, exist_ok=False)
        filename = AUDIO_FILE_PATH + doc + "/full_audio.wav"
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'outtmpl': filename
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([link])
        os.system("youtube-dl --rm-cache-dir")


def trim_audio(id: list, startMin: list, startSec: list, endMin: list, endSec: list) -> None:
    """

    :param id: list of strings representing debates IDs
    :param startMin: list of strings representing the number of minutes to be cut from the beginning of the file
    :param startSec: list of strings representing the number of seconds to be cut from the beginning of the file
    :param endMin: list of strings representing the number of minutes to be cut from the end of the file
    :param endSec: list of strings representing the number of seconds to be cut from the end of the file
    :return None: None. The function removes from the original audio file the portions of audio corresponding
                      to the specified seconds and minutes and saves a new version of the file '_trim.wav' in
                      'files/debates_audio_recordings' (in the corresponding debate's sub folder).
    """
    for i in tqdm(range(len(id))):
        FOLDER_ID = id[i]

        EXPORT_FILENAME = "files/debates_audio_recordings/" + FOLDER_ID + "/full_audio_trim.wav"
        IMPORT_FILENAME = "files/debates_audio_recordings/" + FOLDER_ID + '/' + 'full_audio.wav'

        # importing file from location by giving its path
        sound = AudioSegment.from_file(IMPORT_FILENAME)

        # Selecting Portion we want to cut
        StrtMin = startMin[i]
        StrtSec = startSec[i]
        duration = sound.duration_seconds
        EndMin, EndSec = divmod(duration, 60)
        EndMin = EndMin - endMin[i]
        EndSec = EndSec - endSec[i]

        # Time to milliseconds conversion
        StrtTime = StrtMin * 60 * 1000 + StrtSec * 1000
        EndTime = EndMin * 60 * 1000 + EndSec * 1000
        # print(EndTime)

        # Opening file and extracting portion of it
        extract = sound[StrtTime:EndTime]
        # Saving file in required location
        extract.export(EXPORT_FILENAME, format="wav")  # wav conversion is faster than mp3 conversion


def copy_transcripts(id: list) -> None:
    """

    :param id: list of strings representing debates IDs
    :return: None. The function copies transcripts from the original dataset folder to the 'files/transcripts' folder.
    """
    for el in id:
        if el not in DELETED_ID:
            current_path = 'files/datasets/YesWeCan/ElecDeb60To16/' + el + '.txt'
            transcript_folder = 'files/transcripts/' + el
            os.makedirs(transcript_folder, exist_ok=False)
            dest_path = transcript_folder
            shutil.copy(current_path, dest_path)


def create_plain_text(id: list) -> None:
    """

    :param id: list of strings representing debates IDs
    :return: None. The function creates the plain version of each transcript, saving a new version '_plain.txt'
             in the subdirectory of the corresponding debate. The function creates the plain version of each
             transcript, saving a new version '_plain.txt' in the subdirectory of the corresponding debate.
             In the plain version, speaker information is removed and the text is tokenized by sentences.
             The plain text thus contains one sentence per line.
    """
    for i in range(len(id)):

        FOLDER_ID = id[i]
        FILENAME = "files/transcripts/" + FOLDER_ID + "/" + FOLDER_ID + '.txt'
        NEW_FILENAME = "files/transcripts/" + FOLDER_ID + "/" + FOLDER_ID + '_plain' + '.txt'
        file1 = open(FILENAME, "r")
        lines = file1.readlines()
        new_text = ""
        for line in lines:
            new_line = line[line.index(':') + 2:]  # remove speaker
            sentences = sent_tokenize(new_line)
            for s in sentences:
                new_text += s + '\n'
        file2 = open(NEW_FILENAME, "w")
        file2.write(new_text)
        # file1.close()
        # file2.close()


def generate_chunks(id: list) -> None:
    """

    :param id: list of strings representing debates IDs
    :return: None. The function generates the 20-minute chunks for each debate and saves them in the 'split'
             sub-folders of each debate in 'files/debates_audio_recordings'
    """
    for i in range(len(id)):
        FOLDER_ID = id[i]
        FILENAME = "files/debates_audio_recordings/" + FOLDER_ID + "/full_audio_trim.wav"
        CHUNKS_FOLDER = "files/debates_audio_recordings/" + FOLDER_ID + "/splits/"

        os.makedirs(CHUNKS_FOLDER, exist_ok=False)

        sound = AudioSegment.from_mp3(FILENAME)
        duration = sound.duration_seconds
        cut_sec = 1200
        n_chunks = round(duration / cut_sec)  # we split files in chunks of 20 mins

        for i in tqdm(range(n_chunks)):
            start_sec = cut_sec * i
            end_sec = cut_sec * (i + 1)
            if i == n_chunks - 1:
                end_sec = duration
            start_time = start_sec * 1000
            end_time = end_sec * 1000
            extract = sound[start_time:end_time]
            chunk_filename = CHUNKS_FOLDER + 'split_' + str(i) + '.wav'

            # Saving file in required location
            extract.export(chunk_filename, format="wav")


def generate_empty_transcript_files(id: list) -> None:
    """

    :param id: list of strings representing debates IDs
    :return: None. The function generates as many empty '.txt' files as there are chunks generated for each debate and
             saves them in the 'splits' subdirectory of each debate in the 'files/transcripts' folder
    """
    for j in range(len(id)):
        FOLDER_ID = id[j]
        SPLIT_TRANSCRIPTS_PATH = 'files/transcripts/' + FOLDER_ID + "/" + 'splits/'
        SPLITS_AUDIO_PATH = 'files/debates_audio_recordings/' + FOLDER_ID + '/' + 'splits'
        os.makedirs(SPLIT_TRANSCRIPTS_PATH, exist_ok=False)
        i = 0
        for filename in os.listdir(SPLITS_AUDIO_PATH):
            if filename != '.DS_Store':  # MacOS hidden files check
                parts = filename.split('.')
                txt_file = parts[0] + '.txt'
                open(SPLIT_TRANSCRIPTS_PATH + txt_file, 'w').close()


def run_aeneas(id: list) -> None:
    """

    :param id: list of strings representing debates IDs
    :return: None. For each debate it executes the script to perform the alignment of audio and text.
             The '.json' files resulting from the alignment come in 'files/alignment_results'.
             A subfolder for each debate.
    """

    for i in tqdm(range(len(id))):
        FOLDER_ID = id[i]

        SPLIT_TRANSCRIPTS_PATH = 'files/transcripts/' + FOLDER_ID + "/" + 'splits/'
        SPLITS_AUDIO_PATH = 'files/debates_audio_recordings/' + FOLDER_ID + '/' + 'splits'
        curr_dir = os.getcwd()
        AENEAS_SCRIPT_FOLDER = curr_dir + "/run_aeneas"
        DEST_CLIP_FOLDER = curr_dir + '/files/alignment_results/' + FOLDER_ID
        os.mkdir(DEST_CLIP_FOLDER)
        for filename in os.listdir(SPLITS_AUDIO_PATH):
            if filename != '.DS_Store':
                parts = filename.split('.')
                txt_file = parts[0] + '.txt'
                split_audio_path = SPLITS_AUDIO_PATH + '/' + filename
                split_text_path = SPLIT_TRANSCRIPTS_PATH + txt_file
                copy_command_text = "cp " + split_text_path + " " + AENEAS_SCRIPT_FOLDER
                copy_command_audio = "cp " + split_audio_path + " " + AENEAS_SCRIPT_FOLDER
                os.system(copy_command_text)
                os.system(copy_command_audio)
        os.chdir('run_aeneas')

        AENEAS_COMMAND = "./run.sh"
        os.system(AENEAS_COMMAND)

        for filename in os.listdir(os.getcwd()):
            if filename != '.DS_Store':
                parts = filename.split('.')
                if parts[-1] == 'json':
                    shutil.move(os.getcwd() + '/' + filename, DEST_CLIP_FOLDER)

        for filename in os.listdir(os.getcwd()):
            if filename != '.DS_Store':
                parts = filename.split('.')
                if parts[-1] == 'wav' or parts[-1] == 'txt':
                    os.system("rm " + filename)
        os.chdir('..')


def generate_clips(id: list) -> None:
    """

    :param id: list of strings representing debates IDs
    :return: None. The function generates, for each debate, the audio clips corresponding to each sentence in the
             dataset. The audio files are saved in 'files/audio_clips' in subfolders corresponding to each debate.
             For each debate it creates a new dataset in which the column corresponding to the id of the clips
             is filled with the id of the corresponding generated clip.
    """
    for i in tqdm(range(len(id))):
        FOLDER_ID = id[i]
        DATASET_PATH = 'files/datasets/' + FOLDER_ID + '/dataset.csv'
        DATASET_CLIP_PATH = 'files/datasets/' + FOLDER_ID + '/dataset_clip.csv'
        FULL_AUDIO_PATH = 'files/debates_audio_recordings/' + FOLDER_ID + '/full_audio_trim.wav'
        if not os.path.exists('files/audio_clips'):
            os.makedirs('files/audio_clips')
        AUDIO_CLIPS_PATH = 'files/audio_clips/' + FOLDER_ID
        os.mkdir(AUDIO_CLIPS_PATH)

        # read dataframe with timestamps
        df = pd.read_csv(DATASET_PATH)

        # generate clips
        sound = AudioSegment.from_file(FULL_AUDIO_PATH)
        total_len = df.shape[0]
        for i, row in tqdm(df.iterrows(), total=total_len, position=0):
            start_time = row['NewBegin']
            end_time = row['NewEnd']
            idClip = 'clip_' + str(i)
            if start_time != 'NOT_FOUND':
                start_time = float(row['NewBegin']) * 1000  # sec -> ms conversion
                end_time = float(row['NewEnd']) * 1000  # sec -> ms conversion
                clip_name = AUDIO_CLIPS_PATH + '/' + idClip + '.wav'
                extract = sound[start_time:end_time]
                extract.export(clip_name, format="wav")
                df.at[i, "idClip"] = idClip

        # save new csv
        df.to_csv(DATASET_CLIP_PATH)


def generate_dataset(id: list) -> None:
    """

    :param id: list of strings representing debates IDs
    :return: None. The function generates a new dataset '.csv' for each debate from the original dataset.
            Each new dataset contains 3 new columns corresponding to the new start and end timestamps calculated
            through the alignment with 'aeneas' and the id of the clip corresponding to each sentence.
            The function also saves a 'duplicates.txt' file for each debate, containing the duplicated
            sentences and the number of occurrences.
    """

    for i in tqdm(range(len(id))):
        FOLDER_ID = id[i]

        DIRECTORY_ALIGNMENTS = 'files/alignment_results/' + FOLDER_ID
        FULL_DATASET_PATH = 'files/datasets/YesWeCan/sentence_db_candidate.csv'
        NEW_FILES_PATH = 'files/datasets/' + FOLDER_ID + '/'

        os.mkdir(NEW_FILES_PATH)
        df_debates = pd.read_csv(FULL_DATASET_PATH)

        # count_rows of debate
        count_row_debate = 0
        for i, row in df_debates.iterrows():
            if row['Document'] == FOLDER_ID:
                count_row_debate += 1

        # generate new dataframe
        rows_new_df = []
        for i, row in df_debates.iterrows():
            if row['Document'] == FOLDER_ID:
                rows_new_df.append(row)
        new_df = pd.DataFrame(rows_new_df)

        # set new datasets columns
        new_col_begin = ['NOT_FOUND' for i in range(count_row_debate)]
        new_col_end = ['NOT_FOUND' for i in range(count_row_debate)]
        new_col_id = ['NOT_FOUND' for i in range(count_row_debate)]
        new_df['NewBegin'] = new_col_begin
        new_df['NewEnd'] = new_col_end
        new_df['idClip'] = new_col_id

        count_matches = 0
        matches = []
        count_matches_no_duplicates = 0
        for filename in os.listdir(DIRECTORY_ALIGNMENTS):
            f = os.path.join(DIRECTORY_ALIGNMENTS, filename)
            # checking if it is a file
            if os.path.isfile(f):
                df = pd.read_json(f, orient=str)
                filename = filename.split('.')
                filename = filename[0].split('_')
                split_index = float(filename[-1])
                mul_factor = split_index * 1200.00
                for j, r in tqdm(df.iterrows(), total=df.shape[0], position=0):
                    for i, row in new_df.iterrows():
                        if row['Speech'].strip() == r.fragments['lines'][0].strip():
                            # print(r.fragments['lines'][0].strip())
                            new_df.at[i, "NewBegin"] = round(float(r.fragments['begin']) + mul_factor, 3)
                            # print(round(float(r.fragments['begin'])+mul_factor,3))
                            new_df.at[i, "NewEnd"] = round(float(r.fragments['end']) + mul_factor, 3)
                            if row['Speech'].strip() not in matches:
                                count_matches_no_duplicates += 1
                            count_matches += 1
                            matches.append(row['Speech'].strip())

        a = dict(Counter(matches))
        for k, v in a.items():
            if v > 1:
                print(k, v)

        # save csv
        new_df.to_csv(NEW_FILES_PATH + 'dataset.csv')

        # save files of duplicates for future removal of  those lines from the dataset
        filedup = open(NEW_FILES_PATH + 'duplicates.txt', 'w')

        for k, v in a.items():
            if v > 1:
                line = k + ' : ' + str(v) + '\n'
                filedup.write(line)
        filedup.close()


'''
def generate_clips(id: list) -> None:
    """
    
    :param id: list of strings representing debates IDs
    :return: None. For each debate, it generates the sound clips corresponding to each phrase and creates a new dataset 
             in which the column corresponding to the id of the clips is filled with the 
             id of the corresponding generated clip. 
    """
    for i in tqdm(range(len(id))):
        FOLDER_ID = id[i]

        DATASET_PATH = 'files/datasets/' + FOLDER_ID + '/dataset.csv'
        DATASET_CLIP_PATH = 'files/datasets/' + FOLDER_ID + '/dataset_clip.csv'
        FULL_AUDIO_PATH = 'files/debates_audio_recordings/' + FOLDER_ID + '/full_audio_trim.wav'
        AUDIO_CLIPS_PATH = 'files/audio_clips/' + FOLDER_ID
        os.mkdir(AUDIO_CLIPS_PATH)

        # read dataframe with timestamps
        df = pd.read_csv(DATASET_PATH)

        # generate clips
        sound = AudioSegment.from_file(FULL_AUDIO_PATH)
        total_len = df.shape[0]
        for i, row in tqdm(df.iterrows(), total=total_len, position=0):
            start_time = row['NewBegin']
            end_time = row['NewEnd']
            idClip = 'clip_' + str(i)
            if start_time != 'NOT_FOUND':
                start_time = float(row['NewBegin']) * 1000  # sec -> ms conversion
                end_time = float(row['NewEnd']) * 1000  # sec -> ms conversion
                clip_name = AUDIO_CLIPS_PATH + '/' + idClip + '.wav'
                extract = sound[start_time:end_time]
                extract.export(clip_name, format="wav")
                df.at[i, "idClip"] = idClip

        # save new csv
        df.to_csv(DATASET_CLIP_PATH)
'''


def remove_duplicates(id: list) -> None:
    """

    :param id: list of strings representing debates IDs
    :return: None. The function removes duplicates in the dataset
    """
    for i in tqdm(range(len(id))):
        FOLDER_ID = id[i]
        # print(FOLDER_ID)
        DATASET_CLIP_PATH = 'files/datasets/' + FOLDER_ID + '/dataset_clip.csv'
        DUPLICATES_FILE_PATH = 'files/datasets/' + FOLDER_ID + '/duplicates.txt'
        DATASET_NO_DUP_PATH = 'files/datasets/' + FOLDER_ID + '/dataset_clip_nodup.csv'
        df = pd.read_csv(DATASET_CLIP_PATH)
        df['Component'] = df['Component'].fillna('999')

        dup_file = open(DUPLICATES_FILE_PATH, 'r')
        lines = dup_file.readlines()
        lines_sentences = []

        # get only text
        for el in lines:
            lines_sentences.append(el.split(':')[0].strip())

        lines_component_nan = []
        indexes_component_nan = []
        for i, row in df.iterrows():
            if row['Component'] == '999':
                lines_component_nan.append((row, i))

        for i, row in df.iterrows():
            for line in lines_component_nan:
                if row['Speech'] == line[0]['Speech'] and row['Component'] != '999':
                    indexes_component_nan.append(line[1])

        new_df = df.drop(indexes_component_nan, axis=0)
        indexes_duplicates = []

        duplicates_without_compnull = []
        lines_nan = []
        for el in lines_component_nan:
            lines_nan.append(el[0]['Speech'])

        flag = 0
        for el in lines_sentences:
            for l in lines_nan:
                if el.strip() == l.strip():
                    flag = 1
            if flag == 0:
                duplicates_without_compnull.append(el)
            flag = 0

        for i, row in new_df.iterrows():
            for line in duplicates_without_compnull:
                if row['Speech'].strip() == line.strip():
                    # print(row['Speech'])
                    indexes_duplicates.append(i)
        final_df = new_df.drop(indexes_duplicates, axis=0)

        nan = []
        for i, row in final_df.iterrows():
            if row['Component'] == '999':
                nan.append(i)
        fdf = final_df.drop(nan, axis=0)

        print(fdf.shape)
        fdf.to_csv(DATASET_NO_DUP_PATH)


def remove_not_found(id: list) -> None:
    """

    :param id: list of strings representing debates IDs
    :return: None. The function removes samples marked 'NOT_FOUND', i.e. sentences for which a match with the alignment
             results was not found.
    """
    for i in tqdm(range(len(id))):
        FOLDER_ID = id[i]
        print(FOLDER_ID)
        DATASET_CLIP_PATH = 'files/datasets/' + FOLDER_ID + '/dataset_clip_nodup.csv'
        DATASET_NO_DUP_PATH_NO_NF = 'files/datasets/' + FOLDER_ID + '/dataset_clip_final.csv'
        df = pd.read_csv(DATASET_CLIP_PATH)
        print(df.shape)

        df = df.drop(df[df['idClip'].str.strip().str.match('NOT_FOUND', na=False)].index)
        print(df.shape)

        # save df without duplicates
        df.to_csv(DATASET_NO_DUP_PATH_NO_NF)


def unify_datasets_debates(id: list) -> None:
    """

    :param id: list of strings representing debates IDs
    :return: None. The function combines the datasets created for each debate to create the new dataset MM-ElecDeb60to16
    """
    for i in tqdm(range(len(id))):
        FOLDER_ID = id[i]
        DATASET_NO_DUP_PATH_NO_NF = 'files/datasets/' + FOLDER_ID + '/dataset_clip_final.csv'
        df = pd.read_csv(DATASET_NO_DUP_PATH_NO_NF)
        break

    for i in tqdm(range(1, len(id))):
        FOLDER_ID = id[i]
        print(FOLDER_ID)
        DATASET_NO_DUP_PATH_NO_NF = 'files/datasets/' + FOLDER_ID + '/dataset_clip_final.csv'
        df_1 = pd.read_csv(DATASET_NO_DUP_PATH_NO_NF)
        df = pd.concat([df, df_1])
    print(df.shape)
    df = df.loc[:, ~df.columns.str.match('Unnamed')]

    # save
    FINAL_DATASET_PATH = 'files/datasets/final_dataset/final_dataset.csv'
    df.to_csv(FINAL_DATASET_PATH)

    # compare dimension to original dataframe
    FULL_DATASET_PATH = 'files/datasets/YesWeCan/sentence_db_candidate.csv'
    df_full = pd.read_csv(FULL_DATASET_PATH)
    print("Actual shape: ", df.shape, "Original shape: ", df_full.shape)


def copy_final_csv() -> None:
    """

    :return: None. The function copies the generated dataset into the official 'MM-USElecDeb60to16' folder,
             renaming the file to 'MM-USElecDeb60to16.csv'
    """
    FINAL_CSV_PATH = 'files/datasets/final_dataset/final_dataset.csv'
    DEST_FINAL_CSV = 'files/MM-USElecDeb60to16'
    shutil.copy(FINAL_CSV_PATH, DEST_FINAL_CSV)
    os.rename(DEST_FINAL_CSV + '/final_dataset.csv', 'files/MM-USElecDeb60to16/MM-USElecDeb60to16.csv')


def copy_clips() -> None:
    """

    :return: None. The function copies the clips contained in 'files/audio_clips'
             to the official folder 'MM-USElecDeb60to16'.
    """
    CLIPS_PATH = 'files/audio_clips'
    DEST_FINAL_CLIPS = 'files/MM-USElecDeb60to16/audio_clips'
    shutil.copytree(CLIPS_PATH, DEST_FINAL_CLIPS)
