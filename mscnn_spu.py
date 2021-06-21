import numpy as np 
import tensorflow as tf
from .layer.swem import SWEM

KL = tf.keras.layers


class ExampleModel(tf.keras.Model):
    def __init__(self, config, return_features = False):
        super(MSCNN_SPU, self).__init__()
        filters = 128
        n_classes = 2
        self.cfg = config
        self.return_features = return_features 
        self.conv1 = KL.Conv2D(filters, (3, 300), (1, 1),padding = 'valid',
                                   activation = 'elu')
        self.dropout_dense1 = KL.Dropout(0.5)
        self.dropout_dense2 = KL.Dropout(0.5)
        self.dense = KL.Dense(256, activation = 'relu')
        self.cls = KL.Dense(n_classes, activation = 'softmax', name = 'predict_probs')
        self.concat = KL.Concatenate()
        embedding_matrix = np.load(self.cfg.preprocess.text.embedding_npy)
        vocab_size, vector_size = embedding_matrix.shape 
        embedding_matrix = tf.keras.initializers.Constant(embedding_matrix)
        
        self.embedding = KL.Embedding(vocab_size, vector_size,
                                      embeddings_initializer = embedding_matrix,
                                      trainable = True)

    def call(self, inputs):
        x = conv(x0)
        max_pool = tf.math.reduce_max(x, axis = 1)
        x = self.dropout_dense1(max_pool)
        x = self.dense(x)
        x = self.dropout_dense2(x)
        x = self.cls(x)
        return x


    
