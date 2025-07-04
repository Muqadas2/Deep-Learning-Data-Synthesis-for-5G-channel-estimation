import tensorflow as tf
from tensorflow.keras import layers, Sequential
from .common_blocks import conv_bn_relu, conv_bn_silu, depthwise_block


def CNN_Optim(conv_bn_relu):
    return Sequential([
        layers.Input(shape=(612, 14, 1)),
        conv_bn_relu(1, 4),
        layers.Conv2D(4, 3, padding='same'),
        layers.ReLU(),
        conv_bn_relu(4, 4),
        layers.Conv2D(4, 3, padding='same'),
        layers.ReLU(),
        layers.Conv2D(1, 1)
    ])

def CNN_OptimA(conv_bn_relu):
    return Sequential([
        layers.Input(shape=(612, 14, 1)),
        conv_bn_relu(1, 4, dilation=2),
        layers.Conv2D(4, 3, padding='same'),
        layers.ReLU(),
        conv_bn_relu(4, 4),
        layers.Conv2D(1, 1)
    ])

def CNN_OptimB(conv_bn_relu):
    return Sequential([
        layers.Input(shape=(612, 14, 1)),
        conv_bn_relu(1, 4, dilation=2),
        layers.Conv2D(2, 3, padding='same'),
        layers.ReLU(),
        conv_bn_relu(2, 4),
        layers.Conv2D(1, 1)
    ])

def CNN_OptimC(conv_bn_relu):
    return Sequential([
        layers.Input(shape=(612, 14, 1)),
        conv_bn_relu(1, 2, dilation=2),
        layers.Conv2D(2, 3, padding='same', dilation_rate=2),
        layers.ReLU(),
        conv_bn_relu(2, 4),
        layers.Conv2D(1, 1)
    ])

def Hybrid_CNN_Transformer_TF(Model, layers):
    input_layer = layers.Input(shape=(612, 14, 1))
    x = layers.Conv2D(2, 3, padding='same', dilation_rate=2)(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(2, 3, padding='same', dilation_rate=2)(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(4, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    H, W, C = x.shape[1], x.shape[2], x.shape[3]
    x = layers.Reshape((H * W, C))(x)
    attn_out = layers.MultiHeadAttention(num_heads=2, key_dim=C)(x, x)
    x = layers.Add()([x, attn_out])
    x = layers.LayerNormalization()(x)
    ffn = layers.Dense(2 * C, activation='relu')(x)
    ffn = layers.Dense(C)(ffn)
    x = layers.Add()([x, ffn])
    x = layers.LayerNormalization()(x)
    x = layers.Reshape((H, W, C))(x)
    output = layers.Conv2D(1, 1)(x)
    return Model(inputs=input_layer, outputs=output)
