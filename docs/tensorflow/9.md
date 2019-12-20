# VGGVOX写法

```python
def voicenet(input_x, reuse=False, is_training=True):
    with tf.name_scope('audio_embedding_network'):
        # input_x = tf.layers.batch_normalization(input_x, training=is_training, name='bbn0', reuse=reuse)
        with tf.variable_scope('conv1') as scope:
            conv1_1 = tf.layers.conv2d(input_x, filters=96, kernel_size=[7,7], strides=[2,2], padding='SAME', reuse=reuse, name='cc1')
            conv1_1 = tf.layers.batch_normalization(conv1_1, training=is_training, name='bbn1', reuse=reuse)
            conv1_1 = tf.nn.relu(conv1_1)
            conv1_1 = tf.layers.max_pooling2d(conv1_1, pool_size=[3,3], strides=[2,2], name='mpool1')
            
        with tf.variable_scope('conv2') as scope:
            conv2_1 = tf.layers.conv2d(conv1_1, filters=256, kernel_size=[5,5], strides=[2,2], padding='SAME', reuse=reuse, name='cc2')
            conv2_1 = tf.layers.batch_normalization(conv2_1, training=is_training, name='bbn2', reuse=reuse)
            conv2_1 = tf.nn.relu(conv2_1)
            conv2_1 = tf.layers.max_pooling2d(conv2_1, pool_size=[3,3], strides=[2,2], name='mpool2')


        with tf.variable_scope('conv3') as scope:
            conv3_1 = tf.layers.conv2d(conv2_1, filters=384, kernel_size=[3,3], strides=[1,1], padding='SAME', reuse=reuse, name='cc3_1')
            conv3_1 = tf.layers.batch_normalization(conv3_1, training=is_training, name='bbn3', reuse=reuse)
            conv3_1 = tf.nn.relu(conv3_1)
            
            conv3_2 = tf.layers.conv2d(conv3_1, filters=256, kernel_size=[3,3], strides=[1,1], padding='SAME', reuse=reuse, name='cc3_2')
            conv3_2 = tf.layers.batch_normalization(conv3_2, training=is_training, name='bbn4', reuse=reuse)
            conv3_2 = tf.nn.relu(conv3_2)

            conv3_3 = tf.layers.conv2d(conv3_2, filters=256, kernel_size=[3,3], strides=[1,1], padding='SAME', reuse=reuse, name='cc3_3')
            conv3_3 = tf.layers.batch_normalization(conv3_3, training=is_training, name='bbn5', reuse=reuse)
            conv3_3 = tf.nn.relu(conv3_3)
            conv3_3 = tf.layers.max_pooling2d(conv3_3, pool_size=[5,3], strides=[3,2], name='mpool3')

        with tf.variable_scope('conv4') as scope:
            conv4_3 = tf.layers.conv2d(conv3_3, filters=4096, kernel_size=[9,1], strides=[1,1], padding='VALID', reuse=reuse, name='cc4_1')
            conv4_3 = tf.layers.batch_normalization(conv4_3, training=is_training, name='bbn6', reuse=reuse)
            conv4_3 = tf.nn.relu(conv4_3)
            # conv4_3 = tf.layers.average_pooling2d(conv4_3, pool_size=[1, conv4_3.shape[2]], strides=[1,1], name='apool4')
            conv4_3 = tf.reduce_mean(conv4_3, axis=[1, 2], name='apool4')
    # if is_training:
    #     conv4_3 = tf.nn.dropout(conv4_3, 0.5)

    # flattened = tf.contrib.layers.flatten(conv4_3)
    flattened = tf.nn.l2_normalize(conv4_3)
    features = tf.layers.dense(flattened, 1024, reuse=reuse, name='fc_audio_vgg')

    # if is_training:
    #     features = tf.nn.dropout(features, 0.5)
    # features = tf.contrib.layers.fully_connected(flattened, 1024, reuse=reuse, activation_fn=None, scope='fc_vgg')

    return features
```