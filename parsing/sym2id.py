from __future__ import print_function
from utils import _build_vocab
import sys


if __name__ == '__main__':
  if len(sys.argv) != 2:
    print('usage: python sym2id.py train.gz')
    sys.exit(0)

  vocabs = _build_vocab(sys.argv[1])
  for word, i in vocabs.iteritems():
    print(word, i)
