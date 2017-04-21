
import numpy as np
from random import shuffle
from scipy import stats
import pickle

data = [map(lambda x: float(x), line.split(',')) for line in open('data/wine.data')]
shuffle(data)
label = np.array([int(point[0])-1 for point in data])
data = stats.zscore(data)
for point in data:
  point[0] = 1.0

fd = open('shuffled_data.txt', 'w')
pickle.dump(data, fd)
fd.close()

fl = open('shuffled_label.txt', 'w')
pickle.dump(label, fl)
fl.close()