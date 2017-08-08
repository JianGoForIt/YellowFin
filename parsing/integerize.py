from __future__ import print_function
from utils import open_file, unkify
from traversal import ptb
import sys


def read_vocab(filename):
  word2id = {}
  for line in open_file(filename):
    word, i = line.split()
    word2id[word] = int(i)
  return word2id


def integerize(filename, sym2id):
  data = []
  for i, line in enumerate(open_file(filename)):
    data.extend([sym2id[_] for _ in ptb(line[:-1], sym2id).split() + ['<eos>']])
    if (i + 1) % 40000 == 0:
      yield data
      data = []
  if data:
    yield data


if __name__ == '__main__':
  if len(sys.argv) != 3:
    print('usage: python integerize.py sym2id.gz silver.gz')
    sys.exit(0)

  sym2id = read_vocab(sys.argv[1])
  for data in integerize(sys.argv[2], sym2id):
    print(' '.join([str(x) for x in data]))
