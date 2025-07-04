# models_tf/all_models.py
import tensorflow as tf
from tensorflow.keras import layers, Sequential, Model

def conv_bn_relu(in_c, out_c, kernel_size=3, padding='same', dilation=1):
    return Sequential([
        layers.Conv2D(out_c, kernel_size, padding=padding, dilation_rate=dilation),
        layers.BatchNormalization(),
        layers.ReLU()
    ])

def conv_bn_silu(in_c, out_c, kernel_size=3, padding='same', dilation=1):
    return Sequential([
        layers.Conv2D(out_c, kernel_size, padding=padding, dilation_rate=dilation),
        layers.BatchNormalization(),
        layers.Activation('swish')
    ])

def depthwise_block(in_ch, out_ch, kernel_size=3, padding='same', dilation=1):
    return Sequential([
        layers.DepthwiseConv2D(kernel_size, padding=padding, dilation_rate=dilation),
        layers.ReLU(),
        layers.Conv2D(out_ch, kernel_size=1),
        layers.BatchNormalization(),
        layers.ReLU()
    ])


def CNN_Original():
    return Sequential([
        layers.Conv2D(2, 9, padding='same', input_shape=(612, 14, 1)),
        layers.ReLU(),
        layers.Conv2D(2, 9, padding='same'),
        layers.ReLU(),
        layers.Conv2D(2, 5, padding='same'),
        layers.ReLU(),
        layers.Conv2D(2, 5, padding='same'),
        layers.ReLU(),
        layers.Conv2D(1, 5, padding='same')
    ])

def CNN_Optim():
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

def CNN_Merged():
    return Sequential([
        layers.Input(shape=(612, 14, 1)),
        conv_bn_relu(1, 8, kernel_size=5),
        conv_bn_relu(8, 8, kernel_size=5),
        layers.Conv2D(1, 1)
    ])

def CNN_Depthwise():
    return Sequential([
        layers.Input(shape=(612, 14, 1)),
        depthwise_block(1, 4),
        depthwise_block(4, 4),
        depthwise_block(4, 4),
        depthwise_block(4, 4),
        layers.Conv2D(1, 1)
    ])

def CNN_OptimDilation():
    return Sequential([
        layers.Input(shape=(612, 14, 1)),
        conv_bn_relu(1, 4),
        layers.Conv2D(4, 3, padding='same'),
        layers.ReLU(),
        conv_bn_relu(4, 4),
        layers.Conv2D(1, 1)
    ])

def CNN_OptimA():
    return Sequential([
        layers.Input(shape=(612, 14, 1)),
        conv_bn_relu(1, 4, dilation=2),
        layers.Conv2D(4, 3, padding='same'),
        layers.ReLU(),
        conv_bn_relu(4, 4),
        layers.Conv2D(1, 1)
    ])

def CNN_OptimB():
    return Sequential([
        layers.Input(shape=(612, 14, 1)),
        conv_bn_relu(1, 4, dilation=2),
        layers.Conv2D(2, 3, padding='same'),
        layers.ReLU(),
        conv_bn_relu(2, 4),
        layers.Conv2D(1, 1)
    ])

def CNN_OptimC():
    return Sequential([
        layers.Input(shape=(612, 14, 1)),
        conv_bn_relu(1, 2, dilation=2),
        layers.Conv2D(2, 3, padding='same', dilation_rate=2),
        layers.ReLU(),
        conv_bn_relu(2, 4),
        layers.Conv2D(1, 1)
    ])



def Hybrid_CNN_Transformer_TF():
    input_layer = layers.Input(shape=(612, 14, 1))

    # CNN Feature Extractor (same as CNN_OptimC)
    x = layers.Conv2D(2, 3, padding='same', dilation_rate=2)(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(2, 3, padding='same', dilation_rate=2)(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(4, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)


    # Flatten spatial dims into tokens (B, tokens, C)
    H, W, C = x.shape[1], x.shape[2], x.shape[3]
    x = layers.Reshape((H * W, C))(x)  # Flatten spatial dims: (B, tokens, C)

    # Lightweight Transformer Block
    attn_out = layers.MultiHeadAttention(num_heads=2, key_dim=C)(x, x)
    x = layers.Add()([x, attn_out])
    x = layers.LayerNormalization()(x)

    ffn = layers.Dense(2 * C, activation='relu')(x)
    ffn = layers.Dense(C)(ffn)
    x = layers.Add()([x, ffn])
    x = layers.LayerNormalization()(x)

    # Restore spatial layout
    x = layers.Reshape((H, W, C))(x)
    output = layers.Conv2D(1, 1)(x)

    return Model(inputs=input_layer, outputs=output)



def CNN_OptimC_2():
    return Sequential([
        layers.Input(shape=(612, 14, 1)),
        conv_bn_silu(1, 2, dilation=2),
        layers.Conv2D(2, 3, padding='same', dilation_rate=2),
        layers.Activation('swish'),
        conv_bn_silu(2, 4),
        layers.Conv2D(1, 1)
    ])

def CNN_OptimC_Depthwise():
    return Sequential([
        layers.Input(shape=(612, 14, 1)),
        depthwise_block(1, 2, dilation=2),
        depthwise_block(2, 2, dilation=2),
        depthwise_block(2, 4),
        layers.Conv2D(1, 1)
    ])

def CNN_OptimC_Depthwise_ResMix():
    input_layer = layers.Input(shape=(612, 14, 1))
    x1 = depthwise_block(1, 2)(input_layer)
    x2 = conv_bn_relu(2, 2)(x1)
    res = layers.Conv2D(2, 1)(x1)
    x = layers.add([x2, res])
    x = depthwise_block(2, 4)(x)
    output_layer = layers.Conv2D(1, 1)(x)
    return tf.keras.Model(inputs=input_layer, outputs=output_layer)


import tensorflow as tf
from tensorflow.keras import layers, Model

def Paper_Lightweight_DNN_CIR_FixedInput():
    # Input: (612, 14, 1)
    input_layer = layers.Input(shape=(612, 14, 1))

    # Flatten to match DNN structure
    x = layers.Flatten()(input_layer)  # shape (612*14, )

    # Determine Q = 2^floor(log2(N))
    N = 612 * 14  # 8568
    Q = 2 ** ((N).bit_length() - 1)  # 8192, but too large; we will use a smaller Q
    Q = 512  # practical lightweight

    # Single hidden layer with tanh (as in paper)
    x = layers.Dense(Q, activation='tanh', name="Hidden_tanh")(x)
    # Output layer matching flattened input shape with linear activation
    x = layers.Dense(N, activation='linear', name="Output_linear")(x)
    # Reshape back to (612, 14, 1)
    output_layer = layers.Reshape((612, 14, 1))(x)
    return Model(inputs=input_layer, outputs=output_layer, name="Paper_Lightweight_DNN_CIR_FixedInput")


