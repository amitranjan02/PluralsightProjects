from typing import Dict, Any, Union
import matplotlib.pyplot as plt
import tensorflow as tf

a = tf.add(2,5)
b = tf.multiply(a ,3)

sess = tf.Session()
replace_dict: Dict[Union[object, Any], int] = {a:15}
print(sess.run(b, feed_dict= replace_dict))

plt.subplot(211)
num_list = tf.random_normal([2,20])
out = sess.run(num_list)
x, y = out
plt.scatter(x,y)
#plt.show()
num_list_1 = tf.random_poisson([6,20],[2,20])
out1 = sess.run(num_list_1)
x1, y1 = out1
plt.subplot(212)
plt.scatter(x1, y1, c = 'red',  label = 'red', alpha= 0.35, edgecolors= 'black' )
plt.show()
