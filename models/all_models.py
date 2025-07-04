import tensorflow as tf
from tensorflow.keras import layers, Sequential, Model

# ---------- Common Building Blocks ----------

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


# ---------- Best Performing Models (Top Tier) ----------

def CNN_Original():
    # Baseline model with deep ReLU CNN layers (no batchnorm)
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


def CNN_OptimC_3():
    input_layer = layers.Input(shape=(612, 14, 1))

    # First conv block using Sequential
    conv1 = Sequential([
        layers.Conv2D(4, 3, padding='same', dilation_rate=2),
        layers.BatchNormalization(),
        layers.Activation('swish')
    ])

    # Intermediate conv + GELU activation as Sequential
    conv2 = Sequential([
        layers.Conv2D(4, 3, padding='same', dilation_rate=2),
        layers.Activation('gelu')
    ])

    # Second conv block using Sequential
    conv3 = Sequential([
        layers.Conv2D(8, 3, padding='same'),
        layers.BatchNormalization(),
        layers.Activation('swish')
    ])

    # Apply sequential blocks
    x = conv1(input_layer)
    x = conv2(x)
    x = conv3(x)

    # Output and skip connection
    out = layers.Conv2D(1, 1)(x)
    skip = layers.Conv2D(1, 1)(input_layer)
    final_out = layers.Add()([out, skip])

    return Model(inputs=input_layer, outputs=final_out)


def CNN_OptimC_3():
    input_layer = layers.Input(shape=(612, 14, 1))

    # First conv block using Sequential
    conv1 = Sequential([
        layers.Conv2D(4, 3, padding='same', dilation_rate=2),
        layers.BatchNormalization(),
        layers.Activation('swish')
    ])

    # Intermediate conv + GELU activation as Sequential
    conv2 = Sequential([
        layers.Conv2D(4, 3, padding='same', dilation_rate=2),
        layers.Activation('gelu')
    ])

    # Second conv block using Sequential
    conv3 = Sequential([
        layers.Conv2D(8, 3, padding='same'),
        layers.BatchNormalization(),
        layers.Activation('swish')
    ])

    # Apply sequential blocks
    x = conv1(input_layer)
    x = conv2(x)
    x = conv3(x)

    # Output and skip connection
    out = layers.Conv2D(1, 1)(x)
    skip = layers.Conv2D(1, 1)(input_layer)
    final_out = layers.Add()([out, skip])

    return Model(inputs=input_layer, outputs=final_out)

def CNN_OptimC_3_NoSkip():
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



def Lightweight_DNN_CIR():
    # Best validation loss. Fully connected network, flattened input.
    input_layer = layers.Input(shape=(612, 14, 1))
    x = layers.Flatten()(input_layer)
    x = layers.Dense(512, activation='tanh')(x)
    x = layers.Dense(612 * 14, activation='linear')(x)
    output_layer = layers.Reshape((612, 14, 1))(x)
    return Model(inputs=input_layer, outputs=output_layer)


def CNN_OptimC_2():
    # Best compact CNN. Uses Swish activation + dilation.
    return Sequential([
        layers.Input(shape=(612, 14, 1)),
        conv_bn_silu(1, 2, dilation=2),
        layers.Conv2D(2, 3, padding='same', dilation_rate=2),
        layers.Activation('swish'),
        conv_bn_silu(2, 4),
        layers.Conv2D(1, 1)
    ])


def CNN_Merged():
    # Simple two-layer CNN with wide filters (5x5) and batchnorm.
    return Sequential([
        layers.Input(shape=(612, 14, 1)),
        conv_bn_relu(1, 8, kernel_size=5),
        conv_bn_relu(8, 8, kernel_size=5),
        layers.Conv2D(1, 1)
    ])


def CNN_OptimA():
    # Optimized CNN with dilation in first conv layer.
    return Sequential([
        layers.Input(shape=(612, 14, 1)),
        conv_bn_relu(1, 4, dilation=2),
        layers.Conv2D(4, 3, padding='same'),
        layers.ReLU(),
        conv_bn_relu(4, 4),
        layers.Conv2D(1, 1)
    ])


def CNN_Optim():
    # ReLU CNN with batchnorm, no dilation. Simple and fast.
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


def CNN_OptimB():
    # Smallest CNN variant with some dilation. Good efficiency.
    return Sequential([
        layers.Input(shape=(612, 14, 1)),
        conv_bn_relu(1, 4, dilation=2),
        layers.Conv2D(2, 3, padding='same'),
        layers.ReLU(),
        conv_bn_relu(2, 4),
        layers.Conv2D(1, 1)
    ])


def CNN_OptimC():
    # Very compact CNN with dilated convolutions and batchnorm.
    return Sequential([
        layers.Input(shape=(612, 14, 1)),
        conv_bn_relu(1, 2, dilation=2),
        layers.Conv2D(2, 3, padding='same', dilation_rate=2),
        layers.ReLU(),
        conv_bn_relu(2, 4),
        layers.Conv2D(1, 1)
    ])


# ---------- Mid Tier / Needs Optimization ----------

def Hybrid_CNN_Transformer_TF():
    # CNN feature extractor + MHA + FFN. Latency is high. Try making smaller.
    input_layer = layers.Input(shape=(612, 14, 1))
    x = layers.Conv2D(2, 3, padding='same', dilation_rate=2)(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(2, 3, padding='same', dilation_rate=2)(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(4, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Tokenize spatial dimensions
    H, W, C = x.shape[1], x.shape[2], x.shape[3]
    x = layers.Reshape((H * W, C))(x)

    # Transformer block (very expensive)
    attn_out = layers.MultiHeadAttention(num_heads=2, key_dim=C)(x, x)
    x = layers.Add()([x, attn_out])
    x = layers.LayerNormalization()(x)

    ffn = layers.Dense(2 * C, activation='relu')(x)
    ffn = layers.Dense(C)(ffn)
    x = layers.Add()([x, ffn])
    x = layers.LayerNormalization()(x)

    # Restore shape
    x = layers.Reshape((H, W, C))(x)
    output = layers.Conv2D(1, 1)(x)

    return Model(inputs=input_layer, outputs=output)


# ---------- Poor Accuracy or Broken ----------

def CNN_OptimDilation():
    # Similar to Optim but with no clear benefit. Mediocre val loss.
    return Sequential([
        layers.Input(shape=(612, 14, 1)),
        conv_bn_relu(1, 4),
        layers.Conv2D(4, 3, padding='same'),
        layers.ReLU(),
        conv_bn_relu(4, 4),
        layers.Conv2D(1, 1)
    ])


def CNN_Depthwise():
    # Pure depthwise convolutions. Low param, but weak performance.
    return Sequential([
        layers.Input(shape=(612, 14, 1)),
        depthwise_block(1, 4),
        depthwise_block(4, 4),
        depthwise_block(4, 4),
        depthwise_block(4, 4),
        layers.Conv2D(1, 1)
    ])


def CNN_OptimC_Depthwise():
    # Depthwise variant of OptimC. Underperforms.
    return Sequential([
        layers.Input(shape=(612, 14, 1)),
        depthwise_block(1, 2, dilation=2),
        depthwise_block(2, 2, dilation=2),
        depthwise_block(2, 4),
        layers.Conv2D(1, 1)
    ])


def CNN_OptimC_Depthwise_ResMix():
    # Intended as a residual mix with depthwise blocks. Produces terrible results.
    input_layer = layers.Input(shape=(612, 14, 1))
    x1 = depthwise_block(1, 2)(input_layer)
    x2 = conv_bn_relu(2, 2)(x1)
    res = layers.Conv2D(2, 1)(x1)
    x = layers.add([x2, res])
    x = depthwise_block(2, 4)(x)
    output_layer = layers.Conv2D(1, 1)(x)
    return tf.keras.Model(inputs=input_layer, outputs=output_layer)


def Paper_Lightweight_DNN_CIR_FixedInput():
    # Paper's DNN model. Huge parameter count (8M) with worse accuracy than lightweight DNN.
    input_layer = layers.Input(shape=(612, 14, 1))
    x = layers.Flatten()(input_layer)
    Q = 512  # Reduced from paper's Q = 8192
    x = layers.Dense(Q, activation='tanh', name="Hidden_tanh")(x)
    x = layers.Dense(612 * 14, activation='linear', name="Output_linear")(x)
    output_layer = layers.Reshape((612, 14, 1))(x)
    return Model(inputs=input_layer, outputs=output_layer, name="Paper_Lightweight_DNN_CIR_FixedInput")
