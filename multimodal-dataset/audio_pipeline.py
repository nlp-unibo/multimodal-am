import utils
import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv('files/dictionary.csv', sep=';')
    #df = pd.read_csv('files/debug.csv', sep=';')
    df.columns = ['id', 'link', 'startMin', 'startSec', 'endMin', 'endSec']
    id = df.id
    link = df.link
    startMin = df.startMin
    startSec = df.startSec
    endMin = df.endMin
    endSec = df.endSec

    utils.youtube_download(id, link)
    utils.trim_audio(id, startMin, startSec, endMin, endSec)
    utils.generate_chunks(id)
    utils.generate_clips(id)
    utils.copy_clips()
