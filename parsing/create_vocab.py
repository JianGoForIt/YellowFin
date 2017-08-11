from __future__ import print_function
from bllipparser import Tree
from collections import defaultdict
import gzip, sys

if __name__ == '__main__':
  if len(sys.argv) != 3:
    print('usage: python create_vocab.py train.gz count')
    sys.exit(0)

  threshold = int(sys.argv[2])
  counts = defaultdict(int)
  for line in gzip.open(sys.argv[1], 'rb'):
    for word in Tree(line).tokens():
      counts[word.lower()] += 1

  for w, c in counts.iteritems():
    if c > threshold:
      print(w)
