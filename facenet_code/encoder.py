import tensorflow as tf
import os

from facenet_code import facenet

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# tf.logging.set_verbosity(tf.logging.INFO)
# tf.logging.set_verbosity(tf.logging.WARN)
# tf.logging.set_verbosity(tf.logging.DEBUG)
# tf.logging.set_verbosity(tf.logging.FATAL)


PATH_ENCODE_EMBEDDED = "facenet_code/weights/20180402-114759.pb"


class Encoder:
    def __init__(self):
        self.sess = tf.Session()
        with self.sess.as_default():
            facenet.load_model(PATH_ENCODE_EMBEDDED)

    def generate_embedding(self, face):
        # Get input and output tensors
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

        prewhiten_face = facenet.prewhiten(face.image)

        # Run forward pass to calculate embeddings
        feed_dict = {images_placeholder: [prewhiten_face], phase_train_placeholder: False}
        return self.sess.run(embeddings, feed_dict=feed_dict)[0]
