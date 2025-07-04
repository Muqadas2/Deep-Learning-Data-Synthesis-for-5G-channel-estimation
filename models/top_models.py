import tensorflow as tf
from tensorflow.keras import layers, Sequential, Model
from .common_blocks import conv_bn_relu, conv_bn_silu

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

def CNN_OptimC_2():
    return Sequential([
        layers.Input(shape=(612, 14, 1)),
        conv_bn_silu(1, 2, dilation=2),
        layers.Conv2D(2, 3, padding='same', dilation_rate=2),
        layers.Activation('swish'),
        conv_bn_silu(2, 4),
        layers.Conv2D(1, 1)
    ])

# widen filters slightly
def CNN_OptimC_2_Improved():
    input_layer = layers.Input(shape=(612, 14, 1))

    x = conv_bn_silu(1, 4, dilation=2)(input_layer)
    x = layers.Conv2D(4, 3, padding='same', dilation_rate=2, groups=2)(x)
    x = layers.Activation(tf.nn.gelu)(x)
    x = conv_bn_silu(4, 8)(x)
    x = layers.Conv2D(1, 1)(x)

    # Residual path
    res = layers.Conv2D(1, 1)(input_layer)
    out = layers.Add()([x, res])

    return tf.keras.Model(inputs=input_layer, outputs=out)

def tiny_denoiser_block(filters=4):
    return tf.keras.Sequential([
        layers.Conv2D(filters, 3, padding='same', activation='relu'),
        layers.Conv2D(filters, 3, padding='same', activation='relu'),
        layers.Conv2D(1, 1, padding='same')  # Output: same shape as input
    ])

def CNN_OptimC_2_WithDenoiser():
    input_layer = layers.Input(shape=(612, 14, 1))

    # --- CNN Backbone ---
    x = conv_bn_silu(1, 4, dilation=2)(input_layer)
    x = layers.Conv2D(4, 3, padding='same', dilation_rate=2, groups=2)(x)
    x = layers.Activation(tf.nn.gelu)(x)
    x = conv_bn_silu(4, 8)(x)
    cnn_out = layers.Conv2D(1, 1)(x)

    # --- Residual connection (optional) ---
    res = layers.Conv2D(1, 1)(input_layer)
    base_out = layers.Add()([cnn_out, res])

    # --- Lightweight Denoiser ---
    denoised_out = tiny_denoiser_block(filters=4)(base_out)

    return tf.keras.Model(inputs=input_layer, outputs=denoised_out)

def CNN_OptimC_2_WithDenoiser_PostRefined():
    input_layer = layers.Input(shape=(612, 14, 1))

    # --- CNN Backbone ---
    x = conv_bn_silu(1, 4, dilation=2)(input_layer)
    x = layers.Conv2D(4, 3, padding='same', dilation_rate=2, groups=2)(x)
    x = layers.Activation(tf.nn.gelu)(x)
    x = conv_bn_silu(4, 8)(x)
    cnn_out = layers.Conv2D(1, 1)(x)

    # --- Residual connection ---
    res = layers.Conv2D(1, 1)(input_layer)
    base_out = layers.Add()([cnn_out, res])

    # --- Lightweight Denoiser ---
    denoised_out = tiny_denoiser_block(filters=4)(base_out)

    # --- ⬇️ Post-Processing Refinement Layer ---
    correction = layers.Conv2D(1, kernel_size=1, padding='same')(denoised_out)
    final_out = layers.Add()([denoised_out, correction])

    return tf.keras.Model(inputs=input_layer, outputs=final_out)



def CNN_Merged():
    return Sequential([
        layers.Input(shape=(612, 14, 1)),
        conv_bn_relu(1, 8, kernel_size=5),
        conv_bn_relu(8, 8, kernel_size=5),
        layers.Conv2D(1, 1)
    ])

def CNN_OptimC_3():
    return Sequential([
        layers.Input(shape=(612, 14, 1)),

        # Block 1: Conv + BN + Swish (dilated)
        layers.Conv2D(4, 3, padding='same', dilation_rate=2),
        layers.BatchNormalization(),
        layers.Activation('swish'),

        # Block 2: Conv + GELU (dilated)
        layers.Conv2D(4, 3, padding='same', dilation_rate=2),
        layers.Activation('gelu'),

        # Block 3: Conv + BN + Swish
        layers.Conv2D(8, 3, padding='same'),
        layers.BatchNormalization(),
        layers.Activation('swish'),

        # Output layer
        layers.Conv2D(1, 1)
    ])

def CNN_OptimC_3_Tiny_Relu():
    return tf.keras.Sequential([
        layers.Input(shape=(612, 14, 1)),

        # Block 1: SiLU + Dilation
        conv_bn_silu(1, 2, dilation=2),

        # Block 2: Conv + GELU
        Sequential([
            layers.Conv2D(2, 3, padding='same', dilation_rate=2),
            layers.Activation('relu')
        ]),

        # Block 3: SiLU
        conv_bn_silu(2, 4),

        # Final output layer
        layers.Conv2D(1, 1)
    ])