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


        self.stage_loss = [0 for _ in range(self.total_stages)]#[0] * self.total_stages
        self.total_loss_train = 0
        #self.total_loss_eval = 0

        self.lr = FLAGS.lr
        self.l_decay_rate = FLAGS.l_decay_rate
        self.l_decay_step = FLAGS.l_decay_step

        self.optimizer = FLAGS.optimizer

        self.build_loss()



    def add_first_stage(self):
        with tf.variable_scope('Stage_X'):
            conv1 = tf.layers.conv2d(inputs=self.images,
                                         filters=128,
                                         kernel_size=[9, 9],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='conv1')
            pool1 = tf.layers.max_pooling2d(inputs=conv1,
                                                pool_size=[2, 2],
                                                strides=2,
                                                padding='valid',
                                                name='pool1')
            ########################################################################################
            conv2 = tf.layers.conv2d(inputs = pool1,
                                         filters=128,
                                         kernel_size=[9, 9],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='conv2')
            
            pool2 = tf.layers.max_pooling2d(inputs=conv2,
                                                pool_size=[2, 2],
                                                strides=2,
                                                padding='valid',
                                                name='pool2')
            ########################################################################################
            conv3 = tf.layers.conv2d(inputs = pool2,
                                         filters=128,
                                         kernel_size=[9, 9],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='conv3')
            pool3 = tf.layers.max_pooling2d(inputs=conv3,
                                                pool_size=[2, 2],
                                                strides=2,
                                                padding='valid',
                                                name='pool3')
            ########################################################################################
            conv4 = tf.layers.conv2d(inputs = pool3,
                                         filters=32,
                                         kernel_size=[5, 5],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='conv4')
            self.stage_X_map  = conv4
        ############################################################################################
        with tf.variable_scope('Stage_1'):
            conv5 = tf.layers.conv2d(inputs = conv4,
                                         filters=512,
                                         kernel_size=[9, 9],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='conv5')
            conv6 = tf.layers.conv2d(inputs = conv5,
                                         filters=512,
                                         kernel_size=[1, 1],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='conv6')
            conv7 = tf.layers.conv2d(inputs = conv6,
                                         filters=self.total_joints + 1,
                                         kernel_size=[1, 1],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='conv7')
            self.stage_maps.append(conv7)



    def add_middle_stage(self, stage_index):
        with tf.variable_scope('Stage_' + str(stage_index)):
            middle_stage_input_map = tf.concat([self.stage_maps[stage_index - 2],self.stage_X_map], axis=3)

            mid_conv1 = tf.layers.conv2d(inputs=middle_stage_input_map,
                                             filters=128,
                                             kernel_size=[11, 11],
                                             strides=[1, 1],
                                             padding='same',
                                             activation=tf.nn.relu,
                                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                             name='mid_conv1')
            mid_conv2 = tf.layers.conv2d(inputs=mid_conv1,
                                         filters=128,
                                         kernel_size=[11, 11],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='mid_conv2')
            mid_conv3 = tf.layers.conv2d(inputs=mid_conv2,
                                         filters=128,
                                         kernel_size=[11, 11],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='mid_conv3')
            mid_conv4 = tf.layers.conv2d(inputs=mid_conv3,
                                         filters=128,
                                         kernel_size=[1, 1],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='mid_conv4')
            mid_conv5 = tf.layers.conv2d(inputs=mid_conv4,
                                         filters=self.total_joints + 1,
                                         kernel_size=[1, 1],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='mid_conv5')
            
            self.stage_maps.append(mid_conv5)

    def build_loss(self):
        for i in range(self.total_stages):
            self.stage_loss[i] = tf.nn.l2_loss(self.stage_maps[i] - self.true_hmap_placeholder) / self.batch_size
            # tf.summary.scalar('Stage' + str(i + 1) + '_loss', self.stage_loss[i])

        for i in range(self.total_stages):
            self.total_loss_train += self.stage_loss[i]
            # tf.summary.scalar('Traning total loss: ', self.total_loss_train)
        
        self.global_step = tf.train.get_or_create_global_step()
        self.new_lr = tf.train.exponential_decay(self.lr, global_step=self.global_step, decay_rate=self.l_decay_rate, 
                                                                                    decay_steps=self.l_decay_step)
        # tf.summary.scalar('Global learning rate: ', self.new_lr)

        self.train_op = tf.contrib.layers.optimize_loss(loss=self.total_loss_train, global_step=self.global_step,
                                                        learning_rate=self.new_lr,optimizer=self.optimizer)








