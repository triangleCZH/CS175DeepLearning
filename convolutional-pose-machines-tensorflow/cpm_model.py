import tensorflow as tf
import pickle
from config import FLAGS

class CPM_Model():
    def __init__(self):
        self.total_stages = FLAGS.total_stages
        self.total_joints = FLAGS.total_joints
        self.batch_size = FLAGS.batch_size
        self.image_size = FLAGS.image_size
        self.images = tf.placeholder(dtype=tf.float32,shape=(None, self.image_size, self.image_size, 3))

        self.hmap_size = FLAGS.hmap_size
        self.true_hmap_placeholder = tf.placeholder(dtype=tf.float32,shape=(None, self.hmap_size, self.hmap_size, self.total_joints + 1))
        
        self.stage_maps = []
        
        
        self.add_first_stage()
        for i in range(2, self.total_stages + 1):
            self.add_middle_stage(i)


        self.stage_loss = [0] * self.total_stages
        self.total_loss_train = 0
        self.total_loss_eval = 0

        self.lr = FLAGS.lr
        self.l_decay_rate = FLAGS.l_decay_rate
        self.l_decay_step = FLAGS.l_decay_step

        self.optimizer = FLAGS.optimizer

        self.build_loss()



    def add_first_stage(self):
        with tf.variable_scope('Stage_X'):
            conv1 = tf.layers.conv2d(inputs=self.images,
                                         filters=64,
                                         kernel_size=[3, 3],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='conv1')
            conv2 = tf.layers.conv2d(inputs=conv1,
                                         filters=64,
                                         kernel_size=[3, 3],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='conv2')
            conv3 = tf.layers.conv2d(inputs=conv2,
                                         filters=64,
                                         kernel_size=[3, 3],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='conv3')
            pool1 = tf.layers.max_pooling2d(inputs=conv3,
                                                pool_size=[2, 2],
                                                strides=2,
                                                padding='valid',
                                                name='pool1')
            ########################################################################################
            conv4 = tf.layers.conv2d(inputs = pool1,
                                         filters=128,
                                         kernel_size=[3, 3],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='conv4')
            conv5 = tf.layers.conv2d(inputs = conv4,
                                         filters=128,
                                         kernel_size=[3, 3],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='conv5')
            conv6 = tf.layers.conv2d(inputs = conv5,
                                         filters=128,
                                         kernel_size=[3, 3],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='conv6')
            pool2 = tf.layers.max_pooling2d(inputs=conv6,
                                                pool_size=[2, 2],
                                                strides=2,
                                                padding='valid',
                                                name='pool2')
            ########################################################################################
            conv7 = tf.layers.conv2d(inputs = pool2,
                                         filters=256,
                                         kernel_size=[3, 3],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='conv7')
            conv8 = tf.layers.conv2d(inputs = conv7,
                                         filters=256,
                                         kernel_size=[3, 3],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='conv8')
            conv9 = tf.layers.conv2d(inputs = conv8,
                                         filters=256,
                                         kernel_size=[3, 3],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='conv9')
            conv10 = tf.layers.conv2d(inputs = conv9,
                                         filters=256,
                                         kernel_size=[3, 3],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='conv10')
            conv11 = tf.layers.conv2d(inputs = conv10,
                                         filters=256,
                                         kernel_size=[3, 3],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='conv11')
            pool3 = tf.layers.max_pooling2d(inputs=conv11,
                                                pool_size=[2, 2],
                                                strides=2,
                                                padding='valid',
                                                name='pool3')
            ########################################################################################
            conv12 = tf.layers.conv2d(inputs = pool3,
                                         filters=512,
                                         kernel_size=[5, 5],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='conv12')
            conv13 = tf.layers.conv2d(inputs = conv12,
                                         filters=512,
                                         kernel_size=[5, 5],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='conv13')
            conv14 = tf.layers.conv2d(inputs = conv13,
                                         filters=512,
                                         kernel_size=[5, 5],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='conv14')
            conv15 = tf.layers.conv2d(inputs = conv14,
                                         filters=512,
                                         kernel_size=[5, 5],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='conv15')
            conv16 = tf.layers.conv2d(inputs = conv15,
                                         filters=512,
                                         kernel_size=[5, 5],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='conv16')
            conv17 = tf.layers.conv2d(inputs = conv16,
                                         filters=512,
                                         kernel_size=[5, 5],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='conv17')
            conv18 = tf.layers.conv2d(inputs = conv17,
                                         filters=512,
                                         kernel_size=[5, 5],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='conv18')
            conv19 = tf.layers.conv2d(inputs = conv18,
                                         filters=512,
                                         kernel_size=[5, 5],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='conv19')
            self.stage_X_map  = tf.layers.conv2d(inputs = conv19,
                                         filters=512,
                                         kernel_size=[5, 5],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='stage_X_map')
        ############################################################################################
        with tf.variable_scope('Stage_1'):
            conv1 = tf.layers.conv2d(inputs=self.stage_X_map,
                                     filters=1024,
                                     kernel_size=[9, 9],
                                     strides=[1, 1],
                                     padding='valid',
                                     activation=tf.nn.relu,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     name='conv1')
            conv2 = tf.layers.conv2d(inputs=conv1,
                                     filters=1024,
                                     kernel_size=[1, 1],
                                     strides=[1, 1],
                                     padding='valid',
                                     activation=tf.nn.relu,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     name='conv2')
            conv3 = tf.layers.conv2d(inputs=conv2,
                                     filters=1024,
                                     kernel_size=[1, 1],
                                     strides=[1, 1],
                                     padding='valid',
                                     activation=tf.nn.relu,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     name='conv3')
            conv4 = tf.layers.conv2d(inputs=conv1,
                                     filters=self.total_joints + 1,
                                     kernel_size=[1, 1],
                                     strides=[1, 1],
                                     padding='valid',
                                     activation=None,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     name='conv4')

            self.stage_maps.append(conv4)



    def add_middle_stage(self, stage_index):
        with tf.variable_scope('Stage_' + str(stage_index)):
            middle_stage_input_map = tf.concat([self.stage_maps[stage - 2],self.stage_X_map], axis=3)

            mid_conv1 = tf.layers.conv2d(inputs=middle_stage_input_map,
                                             filters=128,
                                             kernel_size=[9, 9],
                                             strides=[1, 1],
                                             padding='same',
                                             activation=tf.nn.relu,
                                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                             name='mid_conv1')
            mid_conv2 = tf.layers.conv2d(inputs=mid_conv1,
                                         filters=128,
                                         kernel_size=[9, 9],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='mid_conv2')
            mid_conv3 = tf.layers.conv2d(inputs=mid_conv2,
                                         filters=128,
                                         kernel_size=[9, 9],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='mid_conv3')
            mid_conv4 = tf.layers.conv2d(inputs=mid_conv3,
                                         filters=128,
                                         kernel_size=[9, 9],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='mid_conv4')
            mid_conv5 = tf.layers.conv2d(inputs=mid_conv4,
                                         filters=128,
                                         kernel_size=[9, 9],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='mid_conv5')
            mid_conv6 = tf.layers.conv2d(inputs=mid_conv5,
                                         filters=128,
                                         kernel_size=[9, 9],
                                         strides=[1, 1],
                                         padding='valid',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='mid_conv6')
            mid_conv7 = tf.layers.conv2d(inputs=mid_conv6,
                                         filters=128,
                                         kernel_size=[1, 1],
                                         strides=[1, 1],
                                         padding='valid',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='mid_conv7')
            mid_conv8 = tf.layers.conv2d(inputs=mid_conv7,
                                         filters=128,
                                         kernel_size=[1, 1],
                                         strides=[1, 1],
                                         padding='valid',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='mid_conv8')
            mid_conv9 = tf.layers.conv2d(inputs=mid_conv8,
                                         filters=self.total_joints + 1,
                                         kernel_size=[1, 1],
                                         strides=[1, 1],
                                         padding='valid',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='mid_conv9')
            self.stage_maps.append(mid_conv9)

    def build_loss():
        for i in range(self.total_stages):
            self.stage_loss[i] = tf.nn.l2_loss(self.stage_maps[stage] - self.true_hmap_placeholder) / self.batch_size
            tf.summary.scalar('Stage' + str(i + 1) + '_loss', self.stage_loss[i])

        for i in range(self.total_stages):
            self.total_loss += self.stage_loss[i]
        tf.summary.scalar('Traning total loss: ', self.total_loss)

        self.global_step = tf.train.get_or_create_global_step()
        self.new_lr = tf.train.exponential_decay(self.lr,global_step=self.global_step, decay_rate=self.l_decay_rate, 
                                                                                    decay_steps=self.l_decay_step)
        tf.summary.scalar('Global learning rate: ', self.new_lr)

        self.train_op = tf.contrib.layers.optimize_loss(loss=self.total_loss, global_step=self.global_step,
                                                        learning_rate=self.new_lr,optimizer=self.optimizer)








