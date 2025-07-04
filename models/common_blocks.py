from tensorflow.keras import layers, Sequential

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
        layers.Conv2D(out_c, kernel_size=1),
        layers.BatchNormalization(),
        layers.ReLU()
    ])
