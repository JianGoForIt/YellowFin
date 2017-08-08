from __future__ import print_function
from bllipparser import RerankingParser, Tree
from utils import open_file, unkify

import math, sys


def generate_nbest(f):
  nbest = []
  count = 0
  for line in f:
    line = line[:-1]
    if line == '':
      continue
    if count == 0: # the very first
      count = int(line.split()[0])
    elif line.startswith('('):
      nbest.append({'ptb': line})
      count -= 1
      if count == 0:
        yield nbest
        nbest = []


def ptb(line, words):
  t = Tree(line)
  forms = []
  ptb_recurse(t.subtrees()[0], words, forms)
  return ' ' + ' '.join(forms) + ' '


def ptb_recurse(t, words, forms):
  forms.append('(' + t.label)
  for child in t.subtrees():
    if child.is_preterminal():
      token = child.tokens()[0]
      if token.lower() not in words:
        forms.append(unkify(token))
      else:
        forms.append(token.lower())
    else:
      ptb_recurse(child, words, forms)
  forms.append(')' + t.label)


def read_vocab(path):
  words = {}
  for line in open_file(path):
    words[line[:-1]] = len(words)
  return words


def remove_duplicates(nbest):
  new_nbest = []
  for t in nbest:
    good = True
    for new_t in new_nbest:
      if t['seq'] == new_t['seq']:
        good = False
        break
    if good:
      new_nbest.append(t)
  return new_nbest


if __name__ == '__main__':
  if len(sys.argv) != 3 and len(sys.argv) != 4:
    print('usage: python traversal.py vocab.gz gold.gz [nbest.gz]')
    sys.exit(0)

  words = read_vocab(sys.argv[1])
  if len(sys.argv) == 3:
    for line in open_file(sys.argv[2]):
      print(ptb(line[:-1], words))
  else:
    rrp = RerankingParser()
    parser = 'wsj/WSJ-PTB3/parser'
    rrp.load_parser_model(parser)
    for gold, nbest in zip(open_file(sys.argv[2]),
                           generate_nbest(open_file(sys.argv[3]))):
      for tree in nbest:
        tree['seq'] = ptb(tree['ptb'], words)
      nbest = remove_duplicates(nbest)
      gold = Tree(gold)
      print(len(nbest))
      for t in nbest:
        scores = Tree(t['ptb']).evaluate(gold)
        print(scores['gold'], scores['test'], scores['matched'])
        print(t['seq'])
