#coding=gbk
import input_data
import tensorflow as tf
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#输入数据设置
x=tf.placeholder("float",shape=[None,784])
y_=tf.placeholder("float",shape=[None,10])

W=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))
#prediction model
y=tf.nn.softmax(tf.matmul(x,W)+b)

#set train model,also part of tensorflow graph
cross_entropy=-tf.reduce_sum(y+tf.log(y))
train_step=tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
#initialize the session ,and then start session,run train
init=tf.global_variables_initializer()

sess=tf.Session()
sess.run(init)
for i in range(1000):
    batch_xs,batch_ys=mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys})


correct_predition=tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_predition,"float"))

print(sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels}))