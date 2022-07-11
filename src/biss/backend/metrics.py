from tensorflow.keras import backend as K
import tensorflow.keras.metrics as metrics

# Keras variant of dice coefficient
def dice_coef(y_true, y_pred, smooth=0):
	y_true_f = K.flatten(y_true)
	y_pred_f = K.flatten(y_pred)
	intersection = K.sum(y_true_f * y_pred_f)
	return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

# Numpy variant of dice coefficient
def compute_dice_coefficient(y_true, y_pred):
    volume_sum = y_true.sum() + y_pred.sum()
    if volume_sum == 0:
        return np.NaN
    volume_intersect = (y_true & y_pred).sum()
    return 2*volume_intersect / volume_sum

# Define metrics to use
def load_metrics():
    return [
        metrics.Precision(name='precision'),
        metrics.Recall(name='recall'),
        metrics.AUC(name='auc'),
        metrics.AUC(name='prc', curve='PR') # precision-recall curve
        #dice_coef
    ]

