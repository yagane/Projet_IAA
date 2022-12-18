from keras.layers import Conv2D
from keras.layers import Input
from keras.layers import ZeroPadding2D
from keras.layers import BatchNormalization
from keras.layers import ReLU
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import add
from keras.models import Model

def conv_block(x, convs, skip=True):

    count = 0
    for conv in convs:
        if count == (len(convs) - 2) and skip:
            skip_connection = x
            
        if conv['stride'] == 2: x = ZeroPadding2D(((1,0),(1,0)))(x)

        x = Conv2D(conv['filtre'], conv['kernel'], strides=conv['stride'], padding='valid' if conv['stride'] == 2 else 'same',
                   name='conv_' + str(conv['idx']))(x)

        x = BatchNormalization(epsilon=0.001, name='bnorm_' + str(conv['idx']))(x)

        x = ReLU(name='relu_' + str(conv['idx']))(x)

        count += 1

    return add([skip_connection, x]) if skip else x

def galaxy_model():
    input_image = Input(shape=(256, 256, 3))

    x = conv_block(input_image, [{'filtre': 32, 'kernel': 3, 'stride': 1, 'idx': 0},
                                 {'filtre': 64, 'kernel': 3, 'stride': 2, 'idx': 1},
                                 {'filtre': 32, 'kernel': 1, 'stride': 1, 'idx': 2},
                                 {'filtre': 64, 'kernel': 3, 'stride': 1,'idx': 3}])

    x = conv_block(x, [{'filtre': 128, 'kernel': 3, 'stride': 2, 'idx': 5},
                       {'filtre': 64, 'kernel': 1, 'stride': 1, 'idx': 6},
                       {'filtre': 128, 'kernel': 3, 'stride': 1, 'idx': 7}])

    x = conv_block(x, [{'filtre': 128, 'kernel': 3, 'stride': 2, 'idx': 9},
                       {'filtre': 64, 'kernel': 1, 'stride': 1, 'idx': 10},
                       {'filtre': 128, 'kernel': 3, 'stride': 1, 'idx': 11}])

    x = conv_block(x, [{'filtre': 256, 'kernel': 3, 'stride': 2, 'idx': 13},
                       {'filtre': 128, 'kernel': 1, 'stride': 1, 'idx': 14},
                       {'filtre': 256, 'kernel': 3, 'stride': 1, 'idx': 15}])

    x = conv_block(x, [{'filtre': 256, 'kernel': 3, 'stride': 2, 'idx': 17},
                        {'filtre': 128, 'kernel': 1, 'stride': 1, 'idx': 18},
                        {'filtre': 256, 'kernel': 3, 'stride': 1, 'idx': 19}])

    x = conv_block(x, [{'filtre': 256, 'kernel': 3, 'stride': 2, 'idx': 21},
                       {'filtre': 128, 'kernel': 1, 'stride': 1, 'idx': 22},
                       {'filtre': 256, 'kernel': 3, 'stride': 1, 'idx': 23}],skip=False)

    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(10)(x)

    model = Model(input_image, x)
    return model


