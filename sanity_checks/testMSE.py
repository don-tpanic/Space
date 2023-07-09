from sklearn.metrics import mean_squared_error
import numpy as np 


"""
Lesson: sklearn mean_squared_error does averaging over all samples while 
        preserving the y dimension; then the final result is averaged over
        all y dimensions.

        whereas a different way is to average over y dimensions first for 
        each sample, then average over all samples.

        In the example below, we show there is no difference.

        But in this project, if we want to measure how error varies across
        samples, we have to do it the second way.
"""
y_true = [[0.5, 1],[-1, 1],[7, -6]]
y_pred = [[0, 2],[-1, 2],[8, -5]]
mse = mean_squared_error(y_true, y_pred)


def mse(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
        
    # error = np.average((y_true[:, :2] - y_pred[:, :2])**2, axis=0)
    error = np.average((y_true[:, :2] - y_pred[:, :2])**2, axis=1)

    print(error.shape)
    error = np.mean(error)
    print(error)
    return error


mse(y_true, y_pred)