import numpy as np # onehot-encode.py

CLASSES = list(np.array([3, 1, 2]))
# The dataset labels
LABELS = np.array([1, 2, 3, 1, 2, 1, 1, 2, 3])
VALUES = [3, 3, 1, 1, 2, 2]
ONEHOT = np.zeros((len(LABELS), len(CLASSES)))

for idx, value in enumerate(LABELS):
  print("idx:",idx,"value:",value)
  ONEHOT[idx, CLASSES.index(value)] = 1

print("One-hot Encoding:")
print(ONEHOT)

