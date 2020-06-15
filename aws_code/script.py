import tensorflow.compat.v1 as tf
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('pb_file', 'mobile/checkpoints/saved_model.pb', "path to pb file")
flags.DEFINE_string('output_dir', 'mobilev1_serve_pb', 'output dir')

def covert_pb_to_server_model(pb_model_path, export_dir, input_name='input', output_name='output'):
    graph_def = read_pb_model(pb_model_path)
    covert_pb_saved_model(graph_def, export_dir, input_name, output_name)
def read_pb_model(pb_model_path):
    with tf.io.gfile.GFile(pb_model_path, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        return graph_def
def covert_pb_saved_model(graph_def, export_dir, input_name='input:0', output_name='output'):
    builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
    sigs = {}
    with tf.Session(graph=tf.Graph()) as sess:
        #names = [n.name for n in tf.get_default_graph().as_graph_def().node]
        #print("NAMES:", names)
        tf.import_graph_def(graph_def, name="")
        g = tf.get_default_graph()
        input_name = 'input:0'
        #output_name = 'MobilenetV1/Predictions/Softmax:0'
        names = [n.name  for n in graph_def.node]
        print("NAME", names)
        out_nodes = [n.name  for n in graph_def.node if n.op in ('Softmax') or n.op in ('ArgMax')]
        print("OUTPUT NAMES:", out_nodes)
        output_name = out_nodes[0] + ':0'
        input_nodes = [n.name for n in graph_def.node if n.op in ('Placeholder')]
        print('input name:', input_nodes)
        print('output name:', output_name) 
        input_name = input_nodes[0] + ':0'
        print("input node:", tf.get_default_graph().get_tensor_by_name(input_name))
        print("input node shape:",  tf.get_default_graph().get_tensor_by_name(input_name).shape)
        print(type(tf.get_default_graph().get_tensor_by_name(input_name).shape))
        inp = g.get_tensor_by_name(input_name)
        out = g.get_tensor_by_name(output_name)
        sigs[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = \
            tf.saved_model.signature_def_utils.predict_signature_def(
                {"input": inp}, {"output": out})
        builder.add_meta_graph_and_variables(sess,
                                             [tag_constants.SERVING],
                                             signature_def_map=sigs)
        builder.save()
covert_pb_to_server_model(FLAGS.pb_file, FLAGS.output_dir)
#covert_pb_to_server_model('pb_files/inception_v2_converted.pb', 'inception')
