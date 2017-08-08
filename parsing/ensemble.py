from __future__ import print_function
import gzip, sys

def generate_nbest(f):
  nbest = {}
  count = 0
  score = 0
  for line in f:
    line = line[:-1]
    if line == '':
      continue
    if count == 0: # the very first
      count = int(line.split()[0])
    else:
      if line.startswith('('):
        nbest[line] = score
        count -= 1
        if count == 0:
          yield nbest
          nbest = {}
      else:
        score = float(line)


def open_file(path):
  if path.endswith('.gz'):
    return gzip.open(path, 'rb')
  else:
    return open(path, 'r')


if __name__ == '__main__':
  nbests = []
  for scored_nbest in generate_nbest(open_file(sys.argv[1])):
    nbests.append(scored_nbest)

  for argv in sys.argv[2:]:
    for i, scored_nbest in enumerate(generate_nbest(open_file(argv))):
      for tree, score in scored_nbest.iteritems():
        nbests[i][tree] += score

  for nbest in nbests:
    print(sorted(nbest, key=nbest.get)[0])
