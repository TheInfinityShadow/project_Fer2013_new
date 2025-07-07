import tensorflow as tf
from tensorflow.keras import layers


class CBAMLayer(layers.Layer):
    def __init__(self, ratio=8):
        super(CBAMLayer, self).__init__()
        self.ratio = ratio

    def build(self, input_shape):
        # This ensures the variables are created only once when the layer is first called.
        self.channel = input_shape[-1]

        # Channel Attention
        self.shared_dense_one = layers.Dense(self.channel // self.ratio, activation='relu')
        self.shared_dense_two = layers.Dense(self.channel)

        # Spatial Attention
        self.spatial_conv = layers.Conv2D(1, (7, 7), padding='same', activation='sigmoid')

    def call(self, input_tensor):
        # Channel Attention
        avg_pool = layers.GlobalAveragePooling2D()(input_tensor)
        max_pool = layers.GlobalMaxPooling2D()(input_tensor)

        avg = self.shared_dense_one(avg_pool)
        avg = self.shared_dense_two(avg)

        max = self.shared_dense_one(max_pool)
        max = self.shared_dense_two(max)

        channel_attention = layers.Add()([avg, max])
        channel_attention = layers.Activation('sigmoid')(channel_attention)
        channel_attention = layers.Reshape((1, 1, self.channel))(channel_attention)
        x = layers.Multiply()([input_tensor, channel_attention])

        # Spatial Attention
        avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(x, axis=-1, keepdims=True)
        concat = layers.Concatenate(axis=-1)([avg_pool, max_pool])
        spatial_attention = self.spatial_conv(concat)
        x = layers.Multiply()([x, spatial_attention])

        return x
