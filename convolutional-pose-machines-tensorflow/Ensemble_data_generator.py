
import tensorflow as tf
from config import FLAGS

class ensemble_data_generator:
    def __init__(self, img_dir, batch_size, image_size, box_size):
        self.img_dir = img_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.box_size = box_size
        
        
    def next(self):
        def _parser(record):
            keys_to_features = {
                'image': tf.FixedLenFeature([], dtype=tf.string),
                'joint': tf.VarLenFeature(dtype=tf.float32)}

            parsed = tf.parse_single_example(record, keys_to_features)

            image = tf.decode_raw(parsed['image'], tf.uint8)
            image = tf.reshape(image, [self.box_size, self.box_size, 3])
            image = tf.image.resize_images(image, [self.image_size, self.image_size])

            joints = tf.sparse_tensor_to_dense(parsed['joint'])
            joints = tf.reshape(joints, [FLAGS.total_joints, 2])
            joints *= self.image_size / self.box_size

            return image, joints
        filenames = [self.img_dir]
        dataset = tf.data.TFRecordDataset(filenames)


        dataset = dataset.map(_parser)
        dataset = dataset.repeat()
        dataset = dataset.batch(self.batch_size)
        self.iterator = dataset.make_one_shot_iterator()
        next_ele = self.iterator.get_next()


        return next_ele
#         def _parser(record):
#             keys_to_features = {
#                 'image': tf.FixedLenFeature([], dtype=tf.string),
#                 'joint': tf.VarLenFeature(dtype=tf.float32)}

#             parsed = tf.parse_single_example(record, keys_to_features)

#             image = tf.decode_raw(parsed['image'], tf.uint8)
#             image = tf.reshape(image, [self.box_size, self.box_size, 3])
#             image = tf.image.resize_images(image, [self.image_size, self.image_size])

#             joints = tf.sparse_tensor_to_dense(parsed['joint'])
#             joints = tf.reshape(joints, [21, 2])
#             #it need to be equally resize as the image
#             joints *= image_size / box_size

#             return image, joints
#         filenames = [self.img_dir]
#         dataset = tf.data.TFRecordDataset(filenames)

#         # Use `Dataset.map()` to build a pair of a feature dictionary and a label
#         # tensor for each example.
#         dataset = dataset.map(_parser)
#         dataset = dataset.repeat()
#         dataset = dataset.batch(self.batch_size)
#         self.iterator = dataset.make_one_shot_iterator()

#     def next(self):
#         # `features` is a dictionary in which each value is a batch of values for
#         # that feature; `labels` is a batch of labels.
#         image, joints = self.iterator.get_next()

# #         with tf.Session() as sess:
# #             image = sess.run(image)
# #             joints = sess.run(joints)

#         return image, joints

