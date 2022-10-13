import utils
import pandas as pd

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    df = pd.read_csv('files/dictionary.csv', sep=';')
    df.columns = ['id', 'link', 'startMin', 'startSec', 'endMin', 'endSec']
    id = df.id
    link = df.link
    startMin = df.startMin
    startSec = df.startSec
    endMin = df.endMin
    endSec = df.endSec

    utils.youtube_download(id, link)
    utils.trim_audio(id, startMin, startSec, endMin, endSec)
    utils.copy_transcripts(id)
    utils.create_plain_text(id)
    utils.generate_chunks(id)
    utils.generate_empty_transcript_files(id)
    utils.run_aeneas(id)
    utils.generate_dataset(id)
    utils.generate_clips(id)
    utils.remove_duplicates(id)
    utils.remove_not_found(id)
    utils.unify_datasets_debates(id)

    utils.copy_final_csv()
    utils.copy_clips()
