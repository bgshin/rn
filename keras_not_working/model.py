from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Lambda, Dense, Dropout, Input, BatchNormalization, Activation, Add, concatenate, merge, Concatenate
from keras.models import Model
from keras.optimizers import Adam, RMSprop


class g_function():
    def __init__(self, size):
        self.dense = Dense(size, activation="relu")

    def __call__(self, inputs):
        return self.dense(inputs)



def RN():
    g1 = g_function(256)
    g2 = g_function(256)
    g3 = g_function(256)
    g4 = g_function(256)


    img_input_shape = (75, 75, 3)
    img_input = Input(shape=img_input_shape)

    question_input_shape = (11)
    question_input = Input((question_input_shape,))

    z = Conv2D(24, (3, 3), strides=2, padding='same', kernel_initializer='he_normal')(img_input)
    z = BatchNormalization()(z)
    z = Activation('relu')(z)

    z = Conv2D(24, (3, 3), strides=2, padding='same', kernel_initializer='he_normal')(z)
    z = BatchNormalization()(z)
    z = Activation('relu')(z)

    z = Conv2D(24, (3, 3), strides=2, padding='same', kernel_initializer='he_normal')(z)
    z = BatchNormalization()(z)
    z = Activation('relu')(z)

    z = Conv2D(24, (3, 3), strides=5, padding='same', kernel_initializer='he_normal')(z)
    z = BatchNormalization()(z)
    obj_tensor = Activation('relu')(z)
    n = 2
    nxn = n * n

    g_outputs = []
    for i in range(nxn):
        object_i = obj_tensor[:, i % n, i / n, :]  # [num_batch, units]
        for j in range(nxn):
            object_j = obj_tensor[:, j % n, j / n, :]

            g_input = concatenate([object_i, object_j, question_input])

            g_z = g1(g_input)
            g_z = g2(g_z)
            g_z = g3(g_z)
            g_z = g4(g_z)

            # g_z = Dense(256, activation="relu")(g_input)
            # g_z = Dense(256, activation="relu")(g_z)
            # g_z = Dense(256, activation="relu")(g_z)
            # g_z = Dense(256, activation="relu")(g_z)

            g_outputs.append(g_z)

        print i

    f_input = Add()(g_outputs)
    f_z = Dense(256, activation="relu")(f_input)
    f_z = Dense(256, activation="relu")(f_z)
    f_z = Dropout(0.5)(f_z)
    f_output = Dense(10, activation="softmax")(f_z)

    model = Model(inputs=[img_input, question_input], outputs=f_output)
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model





def bn_layer(x, conv_unit):
    def f(inputs):
        md = Conv2D(x, (conv_unit, conv_unit), padding='same', kernel_initializer='he_normal')(inputs)
        md = BatchNormalization()(md)
        return Activation('relu')(md)

    return f


def conv_net(inputs):
    model = bn_layer(24, 3)(inputs)
    model = bn_layer(24, 3)(model)
    model = bn_layer(24, 3)(model)
    model = bn_layer(24, 3)(model)
    return model



def slice_1(t):
    return t[:, 0, :, :]


def slice_2(t):
    return t[:, 1:, :, :]


def slice_3(t):
    return t[:, 0, :]


def slice_4(t):
    return t[:, 1:, :]




def stack_layer(layers):
    def f(x):
        for k in range(len(layers)):
            x = layers[k](x)
        return x
    return f


def get_MLP(n):
    r = []
    for k in range(n):
        s = stack_layer([
            Dense(256),
            BatchNormalization(),
            Activation('relu')
        ])
        r.append(s)
    return stack_layer(r)


def bn_dense(x):
    y = Dense(256)(x)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Dropout(0.5)(y)
    return y

def RN2():
    input1 = Input((75, 75, 3))
    input2 = Input((11,))
    cnn_features = conv_net(input1)
    lstm_encode = input2
    shapes = cnn_features.shape
    w, h = shapes[1], shapes[2]

    slice_layer1 = Lambda(slice_1)
    slice_layer2 = Lambda(slice_2)
    slice_layer3 = Lambda(slice_3)
    slice_layer4 = Lambda(slice_4)

    features = []
    for k1 in range(w):
        features1 = slice_layer1(cnn_features)
        cnn_features = slice_layer2(cnn_features)
        for k2 in range(h):
            features2 = slice_layer3(features1)
            features1 = slice_layer4(features1)
            features.append(features2)

    relations = []
    concat = Concatenate()
    for feature1 in features:
        for feature2 in features:
            relations.append(concat([feature1, feature2, lstm_encode]))

    g_MLP = get_MLP(3)

    mid_relations = []
    for r in relations:
        mid_relations.append(g_MLP(r))
    combined_relation = Add()(mid_relations)

    rn = bn_dense(combined_relation)
    rn = bn_dense(rn)
    pred = Dense(10, activation='softmax')(rn)

    model = Model(inputs=[input1, input2], outputs=pred)
    optimizer = Adam(lr=3e-4)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model