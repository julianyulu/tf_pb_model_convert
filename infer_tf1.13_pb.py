# -*- coding: utf-8 -*-
import os
import pandas as pd 
import tensorflow as tf
#from config import Config
import numpy as np
import re
import pdb
import glob
import jieba
from sklearn.metrics import classification_report 

#pb_model = 'tf1_model/model.pb'

#pb_model = '/data/julianlu/Experiment/multimodal_emotion/tf1_model/model.pb' (not working)

#pb_model = 'tf113_pb_graph/model.pb' # (workds)

pb_model = 'tf115_graph2graph/model.pb' # (workds)

class Conf:
    def __init__(self, filepath):
        self.word_to_id = self.build_dict(filepath)
        self.num_steps = 32

    def build_dict(self, filepath):
        res = {}
        with open(filepath, 'r') as fp:
            for line in fp:
                splits = line.strip().split()
                res[splits[1]] = int(splits[0])
        return res

config = Conf(pb_model.replace('model.pb', 'id_word'))

def prepare_model(pb_file):

    sess = tf.Session()
    with tf.gfile.FastGFile(pb_file, 'rb') as f:
        # 使用tf.GraphDef()定义一个空的Graph
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        graph = sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')
        node_names = [n.name for n in tf.get_default_graph().as_graph_def().node]
        print(node_names)
        print(">>>>>>>>>>>>>>", node_names)
        init_vars = ["init"]
        #init_vars = ["init_all_tables"]
        for init_var in init_vars:
            if init_var in node_names:
                print("="*10)
                print(init_var)
                sess.run(sess.graph.get_operation_by_name(init_var))
        
        # 获取输入输出的tensor_name
        #input_x = sess.graph.get_tensor_by_name('x:0')
        input_x = sess.graph.get_tensor_by_name('seq_wordids:0')
        #input_len = sess.graph.get_tensor_by_name('seq_lengths:0')
        #input_drop = sess.graph.get_tensor_by_name('dropout:0')
        pred_y = sess.graph.get_tensor_by_name('predict_probs:0')
        return sess, input_x, pred_y

sess, input_x, pred_y = prepare_model(pb_model)

def infer(text, label):
    preds = []
    #pdb.set_trace()
    for text_input in text:
        #print('=============')
        #text_input = re.sub('\s+', '', text_input)
        #text_input = list(jieba.cut(text_input))
        #print(text_input, end = '\t')
        text_input = re.split('\s', text_input)

        _word_ids = [config.word_to_id[w] for w in text_input if w in config.word_to_id]
        length = len(_word_ids) if len(_word_ids) <= config.num_steps else config.num_steps
        word_ids = np.zeros(config.num_steps, dtype=np.float32)
        word_ids[: length] = np.asarray(_word_ids, dtype=np.float32)[: length]
        word_ids = np.expand_dims(word_ids, 0)
        #print(word_ids)

        
        probs=sess.run(pred_y,feed_dict={input_x:word_ids})
        
        # probs=sess.run(pred_y,feed_dict={input_x:word_ids,
        #                                  input_drop: 1.0,
        #                                  input_len: [config.num_steps]})  #需要

        probs = probs[0]
        int_label = np.argmax(probs)
        preds.append(int_label)
        #print(probs)
    print(classification_report(label, preds))

if __name__ == '__main__':
    print('done')

    #txt_file = '/data/julianlu/miraclema_sentiment/ecommerce/test/corpus_ecommerce_gold.3652.txt'

    txt_file = 'mix_bal_test.txt'
    with open(txt_file, 'r') as fp:
        lines = fp.readlines()
    text, label  = zip(*[re.split('##', line.strip()) for line in lines])
    label = [int(x) for x in label]
    print(text[0], label[0])
    infer(text, label)
