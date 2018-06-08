import tensorflow as tf

class CPM_Model():
    def __init__(self, input_size, heatmap_size, total_stages, joints):
        self.stages = total_stages
        self.stage_maps = []
        self.stage_loss = [0] * self.stages
        self.joints = joints

        self.input_images = tf.placeholder(dtype=tf.float32,
                                           shape=(None, input_size, input_size, 3))
        
        self.hmap_placeholder = tf.placeholder(dtype=tf.float32,
                                                  shape=(None, heatmap_size, heatmap_size, joints + 1))
        self.first_stage()
        for stage in range(2, self.stages + 1):
            self.add_middle_stage(stage)

    def first_stage(self):
        with tf.variable_scope('sub_stages'):
            sub_conv1 = tf.layers.conv2d(inputs=self.input_images,
                                         filters=64,
                                         kernel_size=[3, 3],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='sub_conv1')

            sub_conv2 = tf.layers.conv2d(inputs=sub_conv1,
                                         filters=64,
                                         kernel_size=[3, 3],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='sub_conv2')
            sub_pool1 = tf.layers.max_pooling2d(inputs=sub_conv2,
                                                pool_size=[2, 2],
                                                strides=2,
                                                padding='valid',
                                                name='sub_pool1')
            sub_conv3 = tf.layers.conv2d(inputs=sub_pool1,
                                         filters=128,
                                         kernel_size=[3, 3],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='sub_conv3')
            sub_conv4 = tf.layers.conv2d(inputs=sub_conv3,
                                         filters=128,
                                         kernel_size=[3, 3],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='sub_conv4')
            sub_pool2 = tf.layers.max_pooling2d(inputs=sub_conv4,
                                                pool_size=[2, 2],
                                                strides=2,
                                                padding='valid',
                                                name='sub_pool2')
            sub_conv5 = tf.layers.conv2d(inputs=sub_pool2,
                                         filters=256,
                                         kernel_size=[3, 3],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='sub_conv5')
            sub_conv6 = tf.layers.conv2d(inputs=sub_conv5,
                                         filters=256,
                                         kernel_size=[3, 3],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='sub_conv6')
            sub_conv7 = tf.layers.conv2d(inputs=sub_conv6,
                                         filters=256,
                                         kernel_size=[3, 3],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='sub_conv7')
            sub_conv8 = tf.layers.conv2d(inputs=sub_conv7,
                                         filters=256,
                                         kernel_size=[3, 3],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='sub_conv8')
            sub_pool3 = tf.layers.max_pooling2d(inputs=sub_conv8,
                                                pool_size=[2, 2],
                                                strides=2,
                                                padding='valid',
                                                name='sub_pool3')
            sub_conv9 = tf.layers.conv2d(inputs=sub_pool3,
                                         filters=512,
                                         kernel_size=[3, 3],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='sub_conv9')
            sub_conv10 = tf.layers.conv2d(inputs=sub_conv9,
                                          filters=512,
                                          kernel_size=[3, 3],
                                          strides=[1, 1],
                                          padding='same',
                                          activation=tf.nn.relu,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          name='sub_conv10')
            sub_conv11 = tf.layers.conv2d(inputs=sub_conv10,
                                          filters=512,
                                          kernel_size=[3, 3],
                                          strides=[1, 1],
                                          padding='same',
                                          activation=tf.nn.relu,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          name='sub_conv11')
            sub_conv12 = tf.layers.conv2d(inputs=sub_conv11,
                                          filters=512,
                                          kernel_size=[3, 3],
                                          strides=[1, 1],
                                          padding='same',
                                          activation=tf.nn.relu,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          name='sub_conv12')
            sub_conv13 = tf.layers.conv2d(inputs=sub_conv12,
                                          filters=512,
                                          kernel_size=[3, 3],
                                          strides=[1, 1],
                                          padding='same',
                                          activation=tf.nn.relu,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          name='sub_conv13')
            sub_conv14 = tf.layers.conv2d(inputs=sub_conv13,
                                          filters=512,
                                          kernel_size=[3, 3],
                                          strides=[1, 1],
                                          padding='same',
                                          activation=tf.nn.relu,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          name='sub_conv14')
            self.stage_x = tf.layers.conv2d(inputs=sub_conv14,
                                                          filters=128,
                                                          kernel_size=[3, 3],
                                                          strides=[1, 1],
                                                          padding='same',
                                                          activation=tf.nn.relu,
                                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                          name='stage_x')

        with tf.variable_scope('stage_1'):
            conv1 = tf.layers.conv2d(inputs=self.stage_x,
                                     filters=512,
                                     kernel_size=[1, 1],
                                     strides=[1, 1],
                                     padding='valid',
                                     activation=tf.nn.relu,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     name='conv1')
            self.stage_maps.append(tf.layers.conv2d(inputs=conv1,
                                                       filters=self.joints+1,
                                                       kernel_size=[1, 1],
                                                       strides=[1, 1],
                                                       padding='valid',
                                                       activation=None,
                                                       kernel_initializer=tf.contrib.layers.xavier_initializer()))
        
                                                   
    def add_middle_stage(self, stage):
        with tf.variable_scope('stage_' + str(stage)):
            input_layer = tf.concat([self.stage_maps[stage - 2],self.stage_x],axis=3)

            Mconv1 = tf.layers.conv2d(inputs=input_layer,
                                         filters=128,
                                         kernel_size=[7, 7],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='Mconv1')
            Mconv2 = tf.layers.conv2d(inputs=Mconv1,
                                         filters=128,
                                         kernel_size=[7, 7],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='Mconv2')
            Mconv3 = tf.layers.conv2d(inputs=Mconv2,
                                         filters=128,
                                         kernel_size=[7, 7],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='Mconv3')
            Mconv4 = tf.layers.conv2d(inputs=Mconv3,
                                         filters=128,
                                         kernel_size=[7, 7],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='Mconv4')
            Mconv5 = tf.layers.conv2d(inputs=Mconv4,
                                         filters=128,
                                         kernel_size=[7, 7],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='Mconv5')
            Mconv6 = tf.layers.conv2d(inputs=Mconv5,
                                         filters=128,
                                         kernel_size=[1, 1],
                                         strides=[1, 1],
                                         padding='valid',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='Mconv6')
            Mconv7 = tf.layers.conv2d(inputs=Mconv6,
                                                    filters=self.joints + 1,
                                                    kernel_size=[1, 1],
                                                    strides=[1, 1],
                                                    padding='valid',
                                                    activation=None,
                                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                    name='Mconv7')
            self.stage_maps.append(Mconv7)

    def build_loss(self, lr, lr_decay_rate, lr_decay_step, optimizer):
        self.total_loss = 0
        self.init_lr = lr
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_step = lr_decay_step
        self.optimizer = optimizer
        self.batch_size = tf.cast(tf.shape(self.input_images)[0], dtype=tf.float32)


        for stage in range(self.stages):
            self.stage_loss[stage] = tf.nn.l2_loss(self.stage_maps[stage] - self.hmap_placeholder,
                                                       name='l2_loss') / self.batch_size

        for stage in range(self.stages):
            self.total_loss += self.stage_loss[stage]

        self.global_step = tf.train.get_or_create_global_step()

        self.cur_lr = tf.train.exponential_decay(self.init_lr,
                                             global_step=self.global_step,
                                             decay_rate=self.lr_decay_rate,
                                             decay_steps=self.lr_decay_step)

        self.train_op = tf.contrib.layers.optimize_loss(loss=self.total_loss,
                                                            global_step=self.global_step,
                                                            learning_rate=self.cur_lr,
                                                            optimizer=self.optimizer)