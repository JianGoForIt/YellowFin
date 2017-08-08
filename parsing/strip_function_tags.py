from __future__ import print_function
from bllipparser import Tree
import fileinput

for line in fileinput.input():
	tree = Tree(line[:-1])
	for subtree in tree.all_subtrees():
		subtree.label_suffix = ''
	print(tree)
