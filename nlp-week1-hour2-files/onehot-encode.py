# https://nlpforhackers.io/deep-learning-introduction/
import numpy as np # onehot-encode.py

CLASSES = list(np.array([3, 1, 2]))
 
# The dataset labels
LABELS = np.array([1, 2, 3, 1, 2, 1, 1, 2, 3])
ONEHOT = np.zeros((len(LABELS), len(CLASSES)))

for idx, value in enumerate(LABELS):
  print("idx:",idx,"value:",value)
  ONEHOT[idx, CLASSES.index(value)] = 1
 
print("One-hot Encoding:")
print(ONEHOT)
 
# [[0. 1. 0.]
#  [0. 0. 1.]
#  [1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]
#  [0. 1. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]
#  [1. 0. 0.]]
 
