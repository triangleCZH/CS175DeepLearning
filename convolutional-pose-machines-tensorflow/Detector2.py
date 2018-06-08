import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import time

from collections import defaultdict
from io import StringIO
#from matplotlib import pyplot as plt
from PIL import Image


tensorflow_path = "/home/qiaohe/models"
sys.path.append(os.getcwd())
sys.path.append(tensorflow_path + "/research")
sys.path.append(tensorflow_path + "/research/object_detection")
sys.path.append(tensorflow_path + "/research/object_detection/utils")


from utils import ops as utils_ops
from utils import label_map_util
from utils import visualization_utils as vis_util
#%matplotlib inline

PATH_TO_CKPT = '/home/qiaohe/models/research/object_detection/face_inference_graph/frozen_inference_graph.pb'
PATH_TO_LABELS = '/home/qiaohe/models/research/object_detection/face2_inference_graph/hand_label_map.pbtxt'
NUM_CLASSES = 1

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

def run_inference_for_single_image(image, graph):
  print("       run_inference_for_single_image starts")
  t0 = time.time()
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
      print("       run_inference_for_single_image takes ", time.time() - t0)
  return output_dict

#PATH_TO_TEST_IMAGES_DIR = 'models/research/object_detection/test_images'
def bounding_box_from_folder(PATH_TO_TEST_IMAGES_DIR, padding, pos=None):
    print("bounding_box_from_folder starts")
    t0 = time.time()
    #load frozen model into memory
    detection_graph = tf.Graph()
    with detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    #image path

    bounding_box = []
    # the input is an image locaion
    if PATH_TO_TEST_IMAGES_DIR.endswith(('png', 'jpg')):
      bounding_box.append(bounding_box_from_file(PATH_TO_TEST_IMAGES_DIR, padding, detection_graph))
    else:
      #print("Our position: ", pos)
      #the input is a folder, with or withour limitation on positions
      file_list = os.listdir(PATH_TO_TEST_IMAGES_DIR)
      if pos != None:
        file_list = file_list[pos[0]:pos[1]]
      #print("File_list: ", file_list)
      for img in file_list:
        bounding_box.append(bounding_box_from_file(os.path.join(PATH_TO_TEST_IMAGES_DIR, img), padding, detection_graph))
    print("bounding_box_from_folder takes ", time.time() - t0)
    return bounding_box

def bounding_box_from_file(PATH_TO_TEST_IMAGES_DIR, padding, detection_graph):
    print("    bounding_box_from_file starts")
    # this section finds one/two bound(s) according to xmin, ymin, xmax, ymax
    #for index, image_path in enumerate(TEST_IMAGE_PATHS):
    #print("You want me to open this image: ", PATH_TO_TEST_IMAGES_DIR)
    t0 = time.time()
    image = Image.open(PATH_TO_TEST_IMAGES_DIR)
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = load_image_into_numpy_array(image)
    height, width, _ = image_np.shape
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection.
    output_dict = run_inference_for_single_image(image_np, detection_graph)
    # Visualization of the results of a detection.
    tmp = output_dict['detection_boxes'][::-1]

    predict_list = []
    if len(output_dict['detection_boxes']) >= 1 and output_dict['detection_scores'][0] > 0.5:
        output = output_dict['detection_boxes'][0,:]
        predict_list.append(adjust_bound_box(output[1] * width, 
          output[0] * height, output[3] * width, output[2] * height, padding, height, width))

        if len(output_dict['detection_boxes']) >= 2 and output_dict['detection_scores'][1] > 0.5:
            output = output_dict['detection_boxes'][1,:]
            predict_list.append(adjust_bound_box(output[1] * width, 
              output[0] * height, output[3] * width, output[2] * height, padding, height, width))
    print("    bounding_box_from_file takes ", time.time() - t0)
    return predict_list

def adjust_bound_box(xmin, ymin, xmax, ymax, padding, img_height, img_width):
  width = xmax - xmin
  height = ymax - ymin
  #xmin, ymin, xmax, ymax
  bounding_box = np.array([xmin - padding, ymin - padding, xmax + padding, ymax + padding])
  delta = height - width
  if delta > 0: #x is less than y
    bounding_box[0] -= delta / 2 #xmin
    bounding_box[2] += delta / 2 #xmax
  else:
    bounding_box[1] += delta / 2 #ymin
    bounding_box[3] -= delta / 2 #ymax

  if bounding_box[3] >= img_height: #ymax
    delta = bounding_box[3] - img_height
    bounding_box[1] -= delta
    bounding_box[3] -= delta
  if bounding_box[2] >= img_width: #xmax
    delta = bounding_box[2] - img_width
    bounding_box[0] -= delta
    bounding_box[2] -= delta 
  if bounding_box[1] < 0: #ymin
    delta = bounding_box[1]
    bounding_box[1] -= delta
    bounding_box[3] -= delta
  if bounding_box[0] < 0: #xmin
    delta = bounding_box[0]
    bounding_box[0] -= delta
    bounding_box[2] -= delta
  bounding_box[bounding_box < 0] = 0 #if the bounding itself is too large that it exceeds picture, then move to up left, and take max(bound, 0)

  bb_box = [int(i) for i in bounding_box]
  if bb_box[3] - bb_box[1] > bb_box[2] - bb_box[0]:
    bb_box[3] -= 1
  elif bb_box[3] - bb_box[1] < bb_box[2] - bb_box[0]:
    bb_box[2] -= 1
  return bb_box

def make_2d_gaussian_map(input_size, half_width_half_maximum=3, center=(0,0)):
    x = np.arange(0, input_size, 1, float)
    y = np.atleast_2d(x).T
    return np.exp(-((x - center[0]) ** 2 + (y - center[1]) ** 2) / 2.0 / half_width_half_maximum / half_width_half_maximum)

def transform_joints_to_heatmap(input_size, heatmap_size, half_width_half_maximum, batch_joints):
    # Generate ground-truth heatmaps from ground-truth 2d joints
    scale = input_size // heatmap_size
    batch_heatmap = []
    for i in range(batch_joints.shape[0]):
        # i is i-th hand
        heatmap = []
        background_heatmap = np.ones(shape=(heatmap_size, heatmap_size))
        for j in range(batch_joints.shape[1]):
            #batch_joins are 2d arrays of 21 * 2
            # 32 * 32
            cur_heatmap = make_2d_gaussian_map(heatmap_size, half_width_half_maximum, center=(batch_joints[i][j] // scale))
            heatmap.append(cur_heatmap)
            # backgrond_heatmap is the backgraound, 32 * 32
            background_heatmap -= cur_heatmap
        # 32 * 32 * 22
        heatmap.append(background_heatmap)
        batch_heatmap.append(heatmap)
    # shape = (num_of_hands, num_of_joints, heatmap_size, heatmap_size)
    # reorder the dimension, in this case ABCD will be ACDB, which is  (num_of_hands, heatmap_size, heatmap_size, num_of_joints)
    batch_heatmap = np.transpose(np.asarray(batch_heatmap), (0, 2, 3, 1))
    return batch_heatmap
