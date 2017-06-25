import tensorflow as tf
import tensorflow.contrib.slim as slim
BatchNormalization = tf.contrib.keras.layers.BatchNormalization
dense = slim.fully_connected
dropout = slim.dropout

class RN(object):
    def __init__(self):
        self.img = tf.placeholder("float", [None, 75, 75, 3])
        self.question = tf.placeholder("float", [None, 11])
        self.answer = tf.placeholder("float", [None, 10])
        self.BATCH_SIZE = tf.shape(self.answer)[0]

        conv1 = self.conv2d(self.img, [75, 75, 3, 24], 1, stride=2)
        conv2 = self.conv2d(conv1, [38, 38, 24, 24], 2, stride=2)
        conv3 = self.conv2d(conv2, [19, 19, 24, 24], 3, stride=2)
        obj_tensor = self.conv2d(conv3, [10, 10, 24, 24], 4, stride=2)

        g_list = []
        n=5
        nxn=n*n
        g_out2 = 0
        for i in range(nxn):
            # 0: (0,0), 1: (1,0), 2: (2,0), 3: (3,0), 4: (4,0), 5: (0,1), 6: (1,1), ... 24: (4,4)
            o1 = obj_tensor[:, i%n, i/n, :]
            for j in range(nxn):
                o2 = obj_tensor[:, j%n, j/n, :]
                if i==0 and j==0:
                    g_out = self.g_function(o1, o2, i, j, self.question, reuse=False)
                else:
                    g_out = self.g_function(o1, o2, i, j, self.question, reuse=True)

                g_list.append(g_out)

        print i

        g_list = tf.stack(g_list, axis=0)
        f_input = tf.reduce_mean(g_list, axis=0)

        f_output = self.f_function(f_input)
        self.logits = tf.nn.softmax(f_output)
        print self.logits

        losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.answer)
        self.loss = tf.reduce_mean(losses)

        # Classification accuracy
        self.pred = tf.argmax(self.logits, 1)
        self.correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.answer, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))


    def g_function(self, o_i, o_j, i, j, q, reuse, scope='g_function'):
        with tf.variable_scope(scope, reuse=reuse):
            coord_i = tf.tile(tf.expand_dims([(i % 5 - 2) / 2., (i / 5 - 2) / 2.], axis=0), [self.BATCH_SIZE, 1])
            coord_j = tf.tile(tf.expand_dims([(j % 5 - 2) / 2., (j / 5 - 2) / 2.], axis=0), [self.BATCH_SIZE, 1])
            g_input = tf.concat([o_i, coord_i, o_j, coord_j, q], axis=1)
            g_1 = dense(g_input, 256)
            g_2 = dense(g_1, 256)
            g_3 = dense(g_2, 256)
            g_4 = dense(g_3, 256)
            return g_4

    def f_function(self,f_input):
        fc_1 = dense(f_input, 256)
        fc_2 = dense(fc_1, 256)
        fc_2 = dropout(fc_2, keep_prob=0.5, is_training=True)
        fc_3 = dense(fc_2, 10, activation_fn=None)
        return fc_3

    def conv2d(self, input, shape, index, stride=2):
        kernel = tf.get_variable('convW%d' % index, shape,
                                 initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32),
                                 dtype=tf.float32, trainable=True)

        with tf.variable_scope('conv%d' % index) as scope:
            conv = tf.nn.conv2d(input, kernel, [1, stride, stride, 1], padding='SAME')
            conv = BatchNormalization()(conv, training=True)
            conv = tf.nn.relu(conv, name=scope.name)

        print conv
        return conv
