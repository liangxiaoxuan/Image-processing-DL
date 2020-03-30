import keras
from keras import layers,Model
from keras.layers import AveragePooling2D, Input,Activation,Conv2D,\
    MaxPool2D,Dense,Activation
from keras.datasets import cifar10


def lenet_v1(X_input, num_classes):

    # Define the input placeholder as a tensor with shape input_shape.
    #X_input = layers.Input(input_shape)


    X = Conv2D(filters=6, kernel_size=5, padding='same',
                      activation='relu')(X_input)
    X = MaxPool2D(strides=(2, 2))(X)
    X = Conv2D(filters=16, kernel_size=5, padding='same',
                      activation='relu')(X)
    X = MaxPool2D(strides=(2, 2))(X)
    final_features = layers.Flatten()(X)
    X = Dense(120)(final_features)
    X = Activation(activation='relu')(X)
    X = Dense(84)(X)
    X = Activation(activation='relu')(X)
    logits = Dense(num_classes)(X)
    outputs = Activation('softmax')(logits)
    #outputs = keras.activations.softmax(axis=-1)(X)

    model = keras.Model(X_input, outputs)

    return model, final_features, logits, outputs


if __name__ == '__main__':
