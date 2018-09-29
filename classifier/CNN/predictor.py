import argparse 
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import json


def load_labels(label_file):
  labels = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    labels.append(l.rstrip())
  return labels


def read_tensor_from_image_file(file_name,
                                input_height=28,
                                input_width=28,
                                input_mean=0,
                                input_std=255,
                                channels=3):
  input_name = "file_reader"
  output_name = "normalized"

  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(
        file_reader, channels=channels, name="png_reader")
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(
        tf.image.decode_gif(file_reader, name="gif_reader"))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
  else:
    image_reader = tf.image.decode_jpeg(file_reader, channels=channels, name="jpeg_reader")

  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0)
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(normalized, [input_mean]), [input_std])
  sess = tf.Session()
  result = sess.run(normalized)
  return result

  # normalized = tf.image.rgb_to_grayscale(resized)
  # sess = tf.Session()
  # result = sess.run(normalized)
  # return result

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it 
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        # tf.import_graph_def(graph_def, name="prefix")
        tf.import_graph_def(graph_def)
    return graph


def predict_from_frozen(image_path, model):
    path_to_models = "../models/"
    with open(path_to_models + 'models.json') as json_data:
      model_data = json.load(json_data)

  
    model_path = path_to_models + model_data[model]['model_file']
    # labels = load_labels(model_data[model]['label_file'])
    input_height = model_data[model]['input_height']
    input_width = model_data[model]['input_width']

    input_node = model_data[model]['input_node']
    output_node = model_data[model]['output_node']

    # model_path = '../models/mnist_model/frozen_model.pb'
    # input_node = 'Reshape'
    # output_node = 'softmax_tensor'

    input_name = "import/" + input_node
    output_name = "import/" + output_node


    graph = load_graph(model_path)

    # We can verify that we can access the list of operations in the graph
    for op in graph.get_operations():
        print(op.name)
        
    input_operation = graph.get_operation_by_name(input_name)
    output_operation = graph.get_operation_by_name(output_name)
    x = input_operation.outputs[0]
    y = output_operation.outputs[0]

    # input parameters are to reshape the image array into a tensor with the right dimensions to feed into neural network
    image = read_tensor_from_image_file(image_path, input_height = input_height, input_width = input_width)
    # print(image)
        
    # We launch a Session
    with tf.Session(graph=graph) as sess:
        # Note: we don't nee to initialize/restore anything
        # There is no Variables in this graph, only hardcoded constants 
        prediction = sess.run(y, feed_dict={
            x: image
        })
        # I taught a neural net to recognise when a sum of numbers is bigger than 45
        # it should return False in this case
        print(prediction) # [[ False ]] Yay, it works!




def predict_from_checkpoint(image_path, model):
    path_to_models = "../models/"
    with open(path_to_models + 'models.json') as json_data:
      model_data = json.load(json_data)
    module = model_data[model]['script']
    checkpoint_path = path_to_models + model_data[model]['checkpoint_file']
    # labels = load_labels(model_data[model]['label_file'])
    input_height = model_data[model]['input_height']
    input_width = model_data[model]['input_width']
    labels = load_labels(path_to_models+model_data[model]['label_file'])

    image = read_tensor_from_image_file(image_path, input_height = input_height, input_width = input_width)

    import importlib
    clf = importlib.import_module(module)

    with tf.Session() as sess:
      checkpoint = tf.train.latest_checkpoint(path_to_models+model)
      saver = tf.train.import_meta_graph(checkpoint_path)
      saver.restore(sess, checkpoint)

      #img = tf.placeholder(shape=[len(data), 28, 28, 1], dtype=tf.float32)
      #data = tf.convert_to_tensor(images, dtype=tf.float32)
      #feed_dict = {"x": pred_data}
      print("model restored")

      emnist_classifier = tf.estimator.Estimator(
        model_fn=clf.cnn_model_fn,
        model_dir=path_to_models+model)

      pred_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": image},
        num_epochs=1,
        shuffle=False)

      pred_results = emnist_classifier.predict(input_fn=pred_input_fn)
      print(list(pred_results))
      # print(labels[(list(pred_results))[0]['classes']-1])


if __name__ == '__main__':

  # image_path = "../output/test/7.png"
  image_path = "../output/test/4.png"
  predict_from_checkpoint(image_path, "emnist_model")



    