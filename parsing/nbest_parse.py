from __future__ import print_function
import fileinput
from bllipparser import RerankingParser, Tree

if __name__ == '__main__':
  rrp = RerankingParser()
  parser = 'wsj/WSJ-PTB3/parser'
  rrp.load_parser_model(parser)
  for line in fileinput.input():
    tokens = Tree(line).tokens()
    nbest = rrp.parse(tokens)
    print(len(nbest))
    for tree in nbest:
      print(tree.ptb_parse)
