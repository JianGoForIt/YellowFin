This is a Wall Street Journal parsing model from the 3rd version of the
Penn Treebank. It is trained on the standard training division (sections
2-21) using section 24 as development.

On WSJ section 23, it gets an f-score of 91.5%. On WSJ section 22,
it gets an f-score of 91.8%.

It was built by DK Choe (dc65@cs.brown.edu).

Differences from other WSJ models
---------------------------------

At this point, there are a number of other WSJ models, each trained on
slightly different versions of WSJ. This one is trained on an unmodified
version of WSJ-PTB3 whereas others may use an AUXified version or David
McClosky's "anydomain" version.

Executive summary: If you don't have a preference, this is the version
of WSJ that you should use.

Technical information
---------------------

Evaluation results from eval-reranker:

# Evaluating second-stage/features/ec50spfinal/test1.gz
# 1333950 features in second-stage/models/ec50spfinal/features.gz
# 1700 sentences in second-stage/features/ec50spfinal/test1.gz
# ncorrect = 28038, ngold = 30633, nparse = 30431, f-score = 0.918315, -log P = 8048.51, 1333950 nonzero features, mean w = 0.000235804, sd w = 0.000470978
# Evaluating second-stage/features/ec50spfinal/test2.gz
# 1333950 features in second-stage/models/ec50spfinal/features.gz
# 2416 sentences in second-stage/features/ec50spfinal/test2.gz
# ncorrect = 40328, ngold = 44276, nparse = 43912, f-score = 0.914592, -log P = 11234.6, 1333950 nonzero features, mean w = 0.000235804, sd w = 0.000470978

Reranker estimator: cvlm-lbfgs
Reranker estimator parameters: l1c10F1n1p2-weights.gz
