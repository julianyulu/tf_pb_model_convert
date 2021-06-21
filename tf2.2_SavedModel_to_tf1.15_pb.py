import os 
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import numpy as np
import pdb

#model_path = '/data/julianlu/Experiment/multimodal_emotion/exp/3cls_bot_bank/pb_model'
model_path = '/data/julianlu/Experiment/multimodal_emotion/exp/3cls_mix_bal_WordEmb_train_L32/pb_model'

max_len = 32

frozen_out_path = 'tf1_model'
frozen_graph_filename = "frozen_graph"
os.makedirs(frozen_out_path, exist_ok = True)

model = tf.keras.models.load_model(model_path)


#Convert Keras model to ConcreteFunction
full_model = tf.function(lambda x: model(x))

# @tf.function()
# def full_model(inputs):
#     pred = model(inputs)
#     return {"predict_probs": pred}

# my_signatures = full_model.get_concrete_function(
#     tf.TensorSpec((None, max_len), tf.int32, name = 'seq_wordids'))# Get frozen ConcreteFunction

# tf.saved_model.save(model, export_dir = frozen_out_path, signatures = my_signatures)


full_model = full_model.get_concrete_function(
    tf.TensorSpec((None, max_len), tf.int32, name = 'seq_wordids'))# Get frozen ConcreteFunction
frozen_func = convert_variables_to_constants_v2(full_model)
frozen_func.graph.as_graph_def()
layers = [op.name for op in frozen_func.graph.get_operations()]
print("-" * 60)
print("Frozen model layers: ")
for layer in layers:
    print(layer)

print("-" * 60)
print("Frozen model inputs: ")
print(frozen_func.inputs)
print("Frozen model outputs: ")
print(frozen_func.outputs)

# Save frozen graph to disk
tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                  logdir=frozen_out_path,
                  name=f"{frozen_graph_filename}.pb",
                  as_text=False)# Save its text representation
tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                  logdir=frozen_out_path,
                  name=f"{frozen_graph_filename}.pbtxt",
                  as_text=True)
