import tensorflow as tf
from tensorflow.python.framework.graph_util import convert_variables_to_constants
#https://zhuanlan.zhihu.com/p/113734249


# load tf15 saved model
#saved_model_dir = 'tf115_saved_model'
saved_model_dir = '/data/julianlu/Experiment/multimodal_emotion/exp_pmu/unnamed_exp/pb_model'
with tf.Session(graph = tf.Graph()) as sess:
    tf.saved_model.loader.load(sess, ['serve'], saved_model_dir)
    graph = tf.get_default_graph()
    feed_dict = {"seq_wordids:0": [[0]* 32]}
    x = sess.run('predict_probs:0', feed_dict = feed_dict)
    print(x)

    out_graph_def = convert_variables_to_constants(
        sess,
        sess.graph_def,
        output_node_names = ['predict_probs'])
    tf.train.write_graph(out_graph_def, 'tf113_pb_graph2', 'model.pb', as_text = False)
    

