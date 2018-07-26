import tensorflow as tf


class BaselineModel(tf.keras.Model):
    def __init__(self, n_hidden, output_dim, scalerX=None, scalerY=None):
        super(BaselineModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(n_hidden, activation=tf.nn.relu)
        self.outputs = tf.keras.layers.Dense(output_dim)
        self.scalerX = scalerX
        self.scalerY = scalerY

    def call(self, inputs, training):
        x = self.dense1(inputs)
        return self.outputs(x)
