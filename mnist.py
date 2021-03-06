# Импортируем dataset MNIST 
from __future__ import print_function
from tensorflow.examples.tutorials.mnist import input_data
from keras.datasets import mnist
mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)


import tensorflow as tf
import numpy as np

n_1 = int(input("Please, enter the number of layers "))
n_2 = int(input("Please, enter the number of neurons "))
number_steps = int(input("Please, enter the number of steps "))

X_train = mnist.train.images
Y_train = mnist.train.labels
X_test = mnist.test.images
Y_test = mnist.test.labels

# Настраиваем параметры
learning_rate = 0.2
batch_size = 128
display_step = 1000
i = 0
# Параметры сети

number_input = 784 
number_classes = 10
i = 0
# Создаем нейронную сеть
def neural_net(x_dict):
    x = x_dict['images']
    l_1 = tf.layers.dense(x, n_1,activation=tf.nn.tanh)
    l_2 = tf.layers.dense(l_1, n_2, activation=tf.nn.tanh)
    out_l = tf.layers.dense(l_2, number_classes,activation=tf.nn.tanh)
    return out_l

# Создаем модель
def model_fn(features, labels, mode):
    logits = neural_net(features)
    pred_classes = tf.argmax(logits, axis=1)
    pred_probas = tf.nn.softmax(logits)

# Оптимизируем модель
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=logits, labels=tf.cast(labels, dtype=tf.int32)))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())
    

# Оцениваем точность модели
    acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)

#Возвращаем значение точности 
    estim = tf.estimator.EstimatorSpec(mode=mode, predictions=pred_classes,loss=loss_op,train_op=train_op, eval_metric_ops={'accuracy': acc_op})
    return estim
  
# Estimator
model = tf.estimator.Estimator(model_fn)

# Тренировка модели
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': mnist.train.images}, y=mnist.train.labels, batch_size=800, num_epochs=10, shuffle=True)
model.train(input_fn, steps=number_steps)
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': mnist.test.images}, y=mnist.test.labels, batch_size=200, shuffle=False)


# Используем метод оценки модели
e = model.evaluate(input_fn)
# Выводим на экран значение точности распознавания

print("Точность модели на тестовой выборке:", e['accuracy']*100)
