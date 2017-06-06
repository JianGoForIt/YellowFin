#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "echo ./prepare.sh wsj-train.gz wsj-dev.gz"
    exit
fi

# One tree per line.
TRAIN=$1
DEV=$2

mkdir wsj
# Remove function tags.
if [[ "$TRAIN" == *.gz ]]
then
   zcat $TRAIN | python strip_function_tags.py | gzip > wsj/x.gz
   zcat $DEV | python strip_function_tags.py | gzip > wsj/y.gz
else
   cat $TRAIN | python strip_function_tags.py | gzip > wsj/x.gz
   cat $DEV | python strip_function_tags.py | gzip > wsj/y.gz
fi

# Download Charniak parser.
python -mbllipparser.ModelFetcher -i WSJ-PTB3 -d wsj

# Generate nbest parses with Charniak parser. On a modern processer, parsing
# section 24 takes about 5 minutes. 
zcat wsj/y.gz | python nbest_parse.py | gzip > wsj/z.gz

# Create a vocab file.
python create_vocab.py wsj/x.gz 9 | gzip > wsj/vocab.gz

# Preprocess train, dev and dev_nbest files.
python traversal.py wsj/vocab.gz wsj/x.gz | gzip > wsj/train.gz
python traversal.py wsj/vocab.gz wsj/y.gz | gzip > wsj/dev.gz
python traversal.py wsj/vocab.gz wsj/y.gz wsj/z.gz | gzip > wsj/dev_nbest.gz

if false; then
    mkdir semi
    python create_vocab.py wsj/x.gz 1 | gzip > semi/vocab.gz

    python traversal.py semi/vocab.gz wsj/x.gz | gzip > semi/train.gz
    python traversal.py semi/vocab.gz wsj/y.gz | gzip > semi/dev.gz
    python traversal.py semi/vocab.gz wsj/y.gz wsj/z.gz | \
	gzip > semi/dev_nbest.gz

    # Path to millions of trees file. One tree per line.
    SILVER='SET THIS PATH'
    python sym2id.py semi/train.gz | gzip > semi/sym2id.gz
    python integerize.py semi/sym2id.gz $SILVER | gzip > semi/silver.gz
fi

# Remove unnecessary data.
rm wsj/[xyz].gz
