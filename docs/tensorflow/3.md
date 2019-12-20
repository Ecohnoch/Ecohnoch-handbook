# 带BN的优化器

```python
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_op2 = tf.train.AdamOptimizer(lr).minimize(loss, var_list=vars)
```