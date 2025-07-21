#!/usr/bin/bash

FILE_PREFIX=$1        # e.g., raw_data_yfcc.tar.gz
OUTPUT_NAME=$2        # e.g., merged_yfcc.tar.gz
CHUNK_START=$3        # e.g., 0
CHUNK_END=$4          # e.g., 8

for ((i=CHUNK_START;i<=CHUNK_END;i++)); do
    IDX=$(printf "%03d" $i)
    cat ${FILE_PREFIX}.${IDX} >> $OUTPUT_NAME
done

echo "Merged file saved as $OUTPUT_NAME"
