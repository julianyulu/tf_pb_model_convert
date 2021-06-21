# -*- coding: utf-8 -*-
#https://zhuanlan.zhihu.com/p/113734249

import os
import tensorflow as tf
from tensorflow.python.saved_model import signature_constants, tag_constants 
pb_model = 'tf1.15_model/model.pb'


builder = tf.saved_model.builder.SavedModelBuilder('tmp')

with tf.gfile.GFile(pb_model, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

sigs = {}
with tf.Session(graph = tf.Graph()) as sess:
    tf.import_graph_def(graph_def, name = "")
    seq_wordids = sess.graph.get_tensor_by_name("seq_wordids:0")
    predict_probs = sess.graph.get_tensor_by_name("predict_probs:0")
    sigs[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = \
        tf.saved_model.signature_def_utils.predict_signature_def(
            {'seq_wordids': seq_wordids},
            {'predict_probs': predict_probs})
    builder.add_meta_graph_and_variables(sess,
                                         [tag_constants.SERVING],
                                         signature_def_map = sigs)
builder.save()
