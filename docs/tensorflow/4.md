# 制作TFRecord

图像+标签：

```python
def _save_tfrecord(self, save: str, preprocessing_func):
    assert os.path.exists(os.path.dirname(save))
    if preprocessing_func == self._default_preprocessing_func_tensor_input:
        preprocessing_func = self._default_preprocessing_func

    writer = tf.python_io.TFRecordWriter(save)
    for ind, (file, label) in enumerate(zip(self.all_images, self.labels)):
        img = preprocessing_func(file, self.config.shape)
        img_raw = img.tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
            'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
        }))
        writer.write(example.SerializeToString())  # Serialize To String
    writer.close()
    print(' TFRecord save success: ', save)
```

可参考的preprocessingfunc:

```python
def _default_preprocessing_func(self, each_img_path, shape_size):
    img = cv2.imread(each_img_path)
    img = img.astype(np.float32)
    if shape_size:
        img = cv2.resize(img, (shape_size, shape_size))
    img = img - 127.5
    img = np.multiply(img, 0.0078125)
    return img
```