from keras.layers import Conv2D
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import ReLU
from keras.layers import ZeroPadding2D
from keras.layers import UpSampling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import add, concatenate
from keras.models import Model

def conv_block(inp, convs, skip=True):
    x = inp
    count = 0
    for conv in convs:
        if count == (len(convs) - 2) and skip:
            skip_connection = x

        if conv['stride'] > 1: x = ZeroPadding2D(((1, 0), (1, 0)))(x)
        x = Conv2D(conv['filter'], conv['kernel'], strides=conv['stride'], padding='valid' if conv['stride'] > 1 else 'same',
                   name='conv_' + str(conv['layer_idx']),
                   use_bias=False if conv['bnorm'] else True)(x)

        if conv['bnorm']:
            x = BatchNormalization(epsilon=0.001, name='bnorm_' + str(conv['layer_idx']))(x)

        if conv['relu']:
            x = ReLU(name='relu_' + str(conv['layer_idx']))(x)

        count += 1

    return add([skip_connection, x]) if skip else x


def galaxy_model():
    input_image = Input(shape=(256, 256, 3))
    # Layer  0 => 4
    x = conv_block(input_image, [{'filter': 32, 'kernel': 3, 'stride': 1, 'bnorm': True, 'relu': True, 'layer_idx': 0},
                                 {'filter': 64, 'kernel': 3, 'stride': 2, 'bnorm': True, 'relu': True, 'layer_idx': 1},
                                 {'filter': 32, 'kernel': 1, 'stride': 1, 'bnorm': True, 'relu': True, 'layer_idx': 2},
                                 {'filter': 64, 'kernel': 3, 'stride': 1, 'bnorm': True, 'relu': True,
                                  'layer_idx': 3}])
    # Layer  5 => 8
    x = conv_block(x, [{'filter': 128, 'kernel': 3, 'stride': 2, 'bnorm': True, 'relu': True, 'layer_idx': 5},
                       {'filter': 64, 'kernel': 1, 'stride': 1, 'bnorm': True, 'relu': True, 'layer_idx': 6},
                       {'filter': 128, 'kernel': 3, 'stride': 1, 'bnorm': True, 'relu': True, 'layer_idx': 7}])
    # Layer  9 => 11
    x = conv_block(x, [{'filter': 64, 'kernel': 1, 'stride': 1, 'bnorm': True, 'relu': True, 'layer_idx': 9},
                       {'filter': 128, 'kernel': 3, 'stride': 1, 'bnorm': True, 'relu': True, 'layer_idx': 10}])
    # Layer 12 => 15
    x = conv_block(x, [{'filter': 256, 'kernel': 3, 'stride': 2, 'bnorm': True, 'relu': True, 'layer_idx': 12},
                       {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'relu': True, 'layer_idx': 13},
                       {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'relu': True, 'layer_idx': 14}])

    # Layer 16 => 19
    x = conv_block(x, [{'filter': 256, 'kernel': 3, 'stride': 2, 'bnorm': True, 'relu': True, 'layer_idx': 16},
                        {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'relu': True, 'layer_idx': 17},
                        {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'relu': True, 'layer_idx': 18}], skip=False)

    # Layer 20 => 23
    x = conv_block(x, [{'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'relu': True, 'layer_idx': 20},
                       {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'relu': True, 'layer_idx': 21},
                       {'filter': 255, 'kernel': 1, 'stride': 1, 'bnorm': False, 'relu': False, 'layer_idx': 22}], skip=False)

    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(10)(x)

    model = Model(input_image, x)
    return model


