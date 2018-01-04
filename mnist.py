# Импортируем dataset MNIST 
from __future__ import print_function
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)

import tensorflow as tf
import numpy as np

number_of_neurons = int(input("Please, enter the number of neurons "))
number_of_layers = int(input("Please, enter the number of layers "))

# Настраиваем параметры
learning_rate = 0.2
number_steps = 20000
batch_size = 128
display_step = 1000

# Параметры сети
n_1 = 300 
n_2 = 300 
number_input = 784 
number_classes = 10 

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
    x={'images': mnist.train.images}, y=mnist.train.labels,
    batch_size=batch_size, num_epochs=None, shuffle=True)
model.train(input_fn, steps=number_steps)
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': mnist.test.images}, y=mnist.test.labels, batch_size=batch_size, shuffle=False)
# Используем метод оценки модели
e = model.evaluate(input_fn)
# Выводим на экран значение точности распознавания
print("Testing Accuracy:", e['accuracy'])
