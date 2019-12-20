# 加载TFRecord

```python
def _load_tfrecord(self, preload: str):
    assert os.path.exists(preload)
    self.inputs = tf.data.TFRecordDataset(preload)
    self.inputs = self.inputs.apply(map_and_batch(self._parse_func, self.config.batch_size, num_parallel_batches=16, drop_remainder=True))

    if self.config.gpu_device:
        self.inputs = self.inputs.apply(prefetch_to_device('/gpu:{}'.format(self.config.gpu_device), None))
    self.iterator = self.inputs.make_one_shot_iterator()
    print(' TFRecord load success: ')
```

# 参考的ParseFunc

```python
def _parse_func(self, example_proto):
    features = {'image_raw': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64)}
    features = tf.parse_single_example(example_proto, features)
    img = tf.decode_raw(features['image_raw'], tf.float32)
    img = tf.reshape(img, shape=(self.config.shape, self.config.shape, 3))
    label = tf.cast(features['label'], tf.int64)
    return img, label
```

