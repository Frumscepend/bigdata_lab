import pandas as pd
import numpy as np
url = r'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
df = pd.read_csv(url, header=None)
df.columns = [u'Чашелистик длина, см',
              u'Чашелистик ширина, см',
              u'Лепесток длина, см',
              u'Лепесток ширина, см',
              'Class']

def parzen_window_est(x_samples, h=1, center=[0,0,0,0,0]):
    dimensions = x_samples.shape[1]
    # print(len(center), dimensions)
    assert (len(center) == dimensions), 'Number of center coordinates have to match sample dimensions'
    k = 0
    for x in x_samples.ix[:, :-1].values:
        is_inside = 1
        for axis,center_point in zip(x, center):
            if np.abs(axis-center_point) > (h/2):
                is_inside = 0
            k += is_inside
    return (k / len(x_samples)) / (h**dimensions)

print('p(x) =', parzen_window_est(df, h=1, center=[5,5,5,5,5]))