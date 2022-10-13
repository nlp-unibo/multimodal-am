#!/bin/bash

NUM_TXT_FILES=$(find . -name '*.txt' | wc -l)

# MAX_RANGE=$NUM_TXT_FILES - 1
START=0
END=$NUM_TXT_FILES
for (( i=$START; i<=$END - 1; i++ ))
do 
    NAME="split_"
    NAME+="$i"
    AUDIOFILE="$NAME"
    AUDIOFILE+=".wav"
    TEXTFILE="$NAME"
    TEXTFILE+=".txt"
    JSONFILE="$NAME"
    JSONFILE+=".json"

    echo $AUDIOFILE

    python -m aeneas.tools.execute_task $AUDIOFILE $TEXTFILE 'task_language=en|os_task_file_format=json|is_text_type=plain' $JSONFILE
done