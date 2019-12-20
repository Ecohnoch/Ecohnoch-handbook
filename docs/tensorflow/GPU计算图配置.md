# Config

```python
config = tf.ConfigProto()
config.allow_soft_placement = True
config.gpu_options.allow_growth = True
```

# 选择GPU

```
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
```

