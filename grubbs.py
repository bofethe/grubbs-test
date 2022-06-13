import numpy as np
from scipy.stats import stats

def grubbs_test(x):
    assert type(x) == np.ndarray, 'Input needs numpy array, but {} was provided'.format(type(x))
    assert x.ndim == 1, 'Input needs 1 demension, but {} was provided'.format(x.ndim)
    
    outliers = []
    def find_outliers(x, outliers):

        mean = np.mean(x)
        median = np.median(x)
        g_calculated = max(abs(x-mean))/np.std(x)

        n = len(x)
        t = stats.t.ppf(1 - 0.05 / (2 * n), n - 2)
        g_critical = ((n - 1) * np.sqrt(np.square(t))) / (np.sqrt(n) * np.sqrt(n - 2 + np.square(t)))

        # check if an outlier is present
        if g_critical < g_calculated:

            # report the max as outlier if positive skew distribution
            if median < mean:
                outlier_i = np.argmax(x)
                outlier_v = x[outlier_i]

            # report the min as outlier if negative skew distribution
            elif median > mean:
                outlier_i = np.argmin(x)
                outlier_v = x[outlier_i]

            outliers.append(outlier_v)
            x = np.delete(x, outlier_i)
            find_outliers(x, outliers)

    find_outliers(x, outliers)   


    if len(outliers) > 0:
        print(len(outliers), 'outliers detected: ', outliers)
        return outliers

    else:
        print('No outliers deteched')
        
test_arr = np.array([1,2,3,4,5,6,200])
t = grubbs_test(test_arr)