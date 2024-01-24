import tensorflow as tf
from tensorflow.keras import layers, models

def transformer_encoder(inputs, num_heads, dff, rate=0.1):
    # Transformer Encoder
    attention_out = layers.MultiHeadAttention(num_heads=num_heads, key_dim=dff)(inputs, inputs)
    attention_out = layers.Dropout(rate)(attention_out)
    out1 = layers.LayerNormalization(epsilon=1e-6)(inputs + attention_out)

    ffn_out = layers.Dense(dff, activation='relu')(out1)
    ffn_out = layers.Dense(inputs.shape[-1])(ffn_out)
    ffn_out = layers.Dropout(rate)(ffn_out)
    out2 = layers.LayerNormalization(epsilon=1e-6)(out1 + ffn_out)

    return out2

def conv_block(inputs, num_filters):
    x = layers.Conv2D(num_filters, 3, padding="same")(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(num_filters, 3, padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    return x

def decoder_block(inputs, skip_features, num_filters):
    x = layers.Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(inputs)
    x = layers.concatenate([x, skip_features])
    x = conv_block(x, num_filters)
    return x

def UNetWithTransformer(input_shape=(256, 256, 3), num_filters=64, num_classes=4):
    inputs = layers.Input(input_shape)

    # Encoder
    x1 = conv_block(inputs, num_filters)
    p1 = layers.MaxPooling2D((2, 2))(x1)
    x2 = conv_block(p1, num_filters * 2)
    p2 = layers.MaxPooling2D((2, 2))(x2)

    # Bottleneck
    b = transformer_encoder(p2, num_heads=4, dff=512)

    # Decoder
    x3 = decoder_block(b, x2, num_filters * 2)
    x4 = decoder_block(x3, x1, num_filters)

    # Output
    outputs = layers.Conv2D(num_classes, (1, 1), padding="same", activation="softmax")(x4)

    model = models.Model(inputs, outputs)
    return model

# Create the model
model = UNetWithTransformer()
model.summary()
