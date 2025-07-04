import tensorflow as tf
from tensorflow.keras import layers, Sequential, Model
from .common_blocks import conv_bn_relu, conv_bn_silu, depthwise_block



def CNN_OptimDilation(conv_bn_relu):
    return Sequential([
        layers.Input(shape=(612, 14, 1)),
        conv_bn_relu(1, 4),
        layers.Conv2D(4, 3, padding='same'),
        layers.ReLU(),
        conv_bn_relu(4, 4),
        layers.Conv2D(1, 1)
    ])

def CNN_Depthwise(depthwise_block):
    return Sequential([
        layers.Input(shape=(612, 14, 1)),
        depthwise_block(1, 4),
        depthwise_block(4, 4),
        depthwise_block(4, 4),
        depthwise_block(4, 4),
        layers.Conv2D(1, 1)
    ])

def CNN_OptimC_Depthwise(depthwise_block):
    return Sequential([
        layers.Input(shape=(612, 14, 1)),
        depthwise_block(1, 2, dilation=2),
        depthwise_block(2, 2, dilation=2),
        depthwise_block(2, 4),
        layers.Conv2D(1, 1)
    ])

def CNN_OptimC_Depthwise_ResMix(tf, conv_bn_relu, depthwise_block):
    input_layer = layers.Input(shape=(612, 14, 1))
    x1 = depthwise_block(1, 2)(input_layer)
    x2 = conv_bn_relu(2, 2)(x1)
    res = layers.Conv2D(2, 1)(x1)
    x = layers.add([x2, res])
    x = depthwise_block(2, 4)(x)
    output_layer = layers.Conv2D(1, 1)(x)
    return tf.keras.Model(inputs=input_layer, outputs=output_layer)

def Paper_Lightweight_DNN_CIR_FixedInput():
    input_layer = layers.Input(shape=(612, 14, 1))
    x = layers.Flatten()(input_layer)
    x = layers.Dense(512, activation='tanh')(x)
    x = layers.Dense(612 * 14, activation='linear')(x)
    output_layer = layers.Reshape((612, 14, 1))(x)
    return Model(inputs=input_layer, outputs=output_layer)
