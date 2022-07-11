from tensorflow.python.keras.layers.core import Dropout
from .metrics import load_metrics
from focal_loss import BinaryFocalLoss
from tensorflow.keras.layers import Conv3D, MaxPooling3D, UpSampling3D, Activation, BatchNormalization, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model, Input

import tensorflow as tf

# TODO: WHERE TO PUT FIX, DEFINE FUNCTION ????
# FIX FOR GTX1080
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


def get_unet3d(input_shape, pool_size=(2, 2, 2), n_labels=1, initial_learning_rate=0.0001,
            depth=2, n_base_filters=32,activation_name="sigmoid", gamma=2):
    """This method generate a unet-3d model for vessel segmentation

    Args:
        input_shape ([type]): [description]
        pool_size (tuple, optional): [description]. Defaults to (2, 2, 2).
        n_labels (int, optional): [description]. Defaults to 1.
        initial_learning_rate (float, optional): [description]. Defaults to 0.0001.
        depth (int, optional): [description]. Defaults to 2.
        n_base_filters (int, optional): [description]. Defaults to 32.
        activation_name (str, optional): [description]. Defaults to "sigmoid".
        gamma (int, optional): [description]. Defaults to 2.

    Returns:
        [type]: [description]
    """

    inputs = Input(input_shape)
    current_layer = inputs
    levels = [] # Store the single levels of UNet architecture

    # ENCODER PATH -> Convolution blocks with max pooling
    for layer_depth in range(depth):
        _n_filters = n_base_filters*(2**layer_depth)

        layer1 = Conv3D(_n_filters, (3, 3, 3), padding='same', strides=(1, 1, 1))(current_layer)
        layer1 = BatchNormalization(axis=1)(layer1)
        layer1 = Activation('relu')(layer1)
        #layer1 = Dropout(0.2)(layer1)

        layer2 = Conv3D(_n_filters*2, (3, 3, 3), padding='same', strides=(1, 1, 1))(layer1)
        layer2 = BatchNormalization(axis=1)(layer2)
        layer2 = Activation('relu')(layer2)
        #layer2 = Dropout(0.2)(layer2)

        if layer_depth < depth - 1:
            current_layer = MaxPooling3D(pool_size=pool_size)(layer2)
            levels.append([layer1, layer2, current_layer])
        else:
            current_layer = layer2
            levels.append([layer1, layer2])

    # DECODER PATH -> Do up-sampling
    for layer_depth in range(depth-2, -1, -1):
        _n_filters = levels[layer_depth][1].shape[-1]

        up_layer = UpSampling3D(size=pool_size)(current_layer)
        concat = concatenate([up_layer, levels[layer_depth][1]], axis=4)

        current_layer = Conv3D(_n_filters, (3, 3, 3), padding='same', strides=(1, 1, 1))(concat)
        current_layer = BatchNormalization(axis=1)(current_layer)
        current_layer = Activation('relu')(current_layer)
        #current_layer = Dropout(0.2)(current_layer)

        current_layer = Conv3D(_n_filters, (3, 3, 3), padding='same', strides=(1, 1, 1))(current_layer)
        current_layer = BatchNormalization(axis=1)(current_layer)
        current_layer = Activation('relu')(current_layer)        
        #current_layer = Dropout(0.2)(current_layer)                
    
    # Define final convolution and activation
    final_convolution = Conv3D(n_labels, (1, 1, 1))(current_layer)
    act = Activation(activation_name)(final_convolution)

    model = Model(inputs=inputs, outputs=act)
    model._name = "UNet_BISS"

    # Define loss function
    _loss = BinaryFocalLoss(gamma=gamma)

    # Compile model
    model.compile(optimizer=Adam(learning_rate=initial_learning_rate), loss=_loss, metrics=load_metrics())
    return model