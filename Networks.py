import os
import numpy as np
import tensorflow as tf

class EF_CNN():
    def __init__(self, args):
        super(EF_CNN, self).__init__()
        self.args = args

    def build_Mabel_Arch(self, input_data, reuse = False, name="Mabel_FE"):

        Layers = []
        with tf.variable_scope(name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse is False

            Layers.append(self.general_conv2d(input_data, 128, 3, stride = 1, padding = 'SAME', activation_function = 'relu', do_norm = False, name = name + '_conv2d_1'))
            Layers.append(tf.layers.max_pooling2d(Layers[-1], 2, 2, name = name + '_maxpooling_1'))
            Layers.append(self.general_conv2d(Layers[-1], 256,  3, stride = 1, padding = 'SAME', activation_function = 'relu', do_norm = False, name = name + '_conv2d_2'))
            Layers.append(tf.layers.max_pooling2d(Layers[-1], 2, 2, name = name + '_maxpooling_2'))
            Layers.append(self.general_conv2d(Layers[-1], 512,  3, stride = 1, padding = 'SAME', activation_function = 'relu', do_norm = False, name = name + '_conv2d_3'))
            #Layers.append(self.resnet_block(Layers[-1], 512, 3, s = 1, name = name + '_resnet_block_1'))
            #Layers.append(self.resnet_block(Layers[-1], 512, 3, s = 1, name = name + '_resnet_block_2'))
            #Layers.append(self.resnet_block(Layers[-1], 512, 3, s = 1, name = name + '_resnet_block_3'))

            #Layers.append(tf.reduce_mean(Layers[-1], axis = [1,2]))
            #Layers.append(tf.layers.flatten(Layers[-1], name = name + '_flatten_1'))

            return Layers

    def build_Ganin_Arch(self, input_data, reuse = False, name = "Ganin_FE"):

        Layers = []
        with tf.variable_scope(name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse is False
            Layers.append(self.general_conv2d(input_data, 32, 5, stride = 1, padding = 'SAME', activation_function = 'relu', do_norm = False, name = name + '_conv2d_1'))
            Layers.append(tf.layers.max_pooling2d(Layers[-1], 2, 2, name = name + '_maxpooling_1'))
            Layers.append(self.general_conv2d(Layers[-1], 48,  5, stride = 1, padding = 'SAME', activation_function = 'relu', do_norm = False, name = name + '_conv2d_2'))
            Layers.append(tf.layers.max_pooling2d(o_c2, 2, 2, name = name + '_maxpooling_2'))
            Layers.append(tf.layers.flatten(Layers[-1], name = name + '_flatten_1'))

            return Layers

    def build_MLP_1hidden_cl(self, input_data, reuse = False, name="MLP"):

        Layers = []
        with tf.variable_scope(name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse is False

            Layers.append(tf.layers.dropout(input_data, 0.2, name= name + '_droput_1'))
            Layers.append(tf.layers.dense(Layers[-1], self.args.num_classes, name = name + '_prediction'))
            Layers.append(tf.nn.softmax(Layers[-1], name = name + '_softmax'))
            return Layers

    def general_conv2d(self, input_data, filters = 64,  kernel_size = 7, stride = 1, stddev = 0.02, activation_function = "relu", padding = "VALID", do_norm=True, relu_factor = 0, name="conv2d"):
        with tf.variable_scope(name):
            conv = tf.layers.conv2d(input_data, filters, kernel_size, stride, padding, activation=None)

            if do_norm:
                conv = tf.layers.batch_normalization(conv, momentum=0.9)

            if activation_function == "relu":
                conv = tf.nn.relu(conv, name = 'relu')
            if activation_function == "leakyrelu":
                conv = tf.nn.leaky_relu(conv, alpha=relu_factor)
            if activation_function == "elu":
                conv = tf.nn.elu(conv, name = 'elu')

            return conv

    def resnet_block(self, input_data, filters, ks = 3, s = 1, name = 'resnet'):

        with tf.variable_scope(name):
            p = int((ks - 1)/2)
            out_res = tf.pad(input_data, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            out_res = self.general_conv2d(input_data, filters, ks, s, 0.02, activation_function = "relu", padding = "VALID", do_norm = True, name = name + '_conv1')
            out_res = tf.pad(out_res,[[0, 0],[p, p],[p, p],[0, 0]], "REFLECT")
            out_res = self.general_conv2d(out_res, filters, ks, s, 0.02, activation_function = None, padding = "SAME", do_norm = True, name = name + '_conv2')
            return  tf.math.add(out_res, input_data, name = 'add_shorcut')

class Domain_Regressors():
    def __init__(self, args):
        super(Domain_Regressors, self).__init__()
        self.args = args
    #=============================GABRIEL: DOMAIN_CLASSIFIER=============================
    def build_Domain_Classifier_Arch(self, input_data, name = "Domain_Classifier_Arch"):
        Layers = []
        with tf.variable_scope(name):
            #Domain Classifier Definition: 2x (Fully_Connected_1024_units + ReLu) + Fully_Connected_1_unit + Logistic

            Layers.append(tf.layers.flatten(input_data))
            Layers.append(self.general_dense(Layers[-1], units=1024, activation_function="relu", name=name + '_dense1'))
            Layers.append(self.general_dense(Layers[-1], units=1024, activation_function="relu", name=name + '_dense2'))
            #SERÁ AQUI O ERRO??? ACTIVATION NONE NA VERDADE USA UMA ATIVAÇÃO LINEAR
            Layers.append(tf.layers.dense(Layers[-1], units=self.args.num_classes, activation=None))
            #ALÉM DISSO, AQUI EU USO UMA SOFTMAX, MAS EM MODELS, OUTRA SOFTMAX É USADA EM CIMA DESSE LOGITS (essa saída não é usada em models)
            Layers.append(tf.nn.softmax(Layers[-1], name=name + '_softmax'))

            return Layers

    def build_Dense_Domain_Classifier(self, input_data, name = "Domain_Classifier_Arch"):
        Layers = []
        num_filters = input_data.get_shape().as_list()[3]
        with tf.variable_scope(name):
            for i in range(3):
                if i == 0:
                    Layers.append(self.general_conv2d(input_data, num_filters/(2**i), 3, stride=1, padding='SAME', activation_function='leakyrelu', do_norm=True, name=name + '_conv2d_' + str(i)))
                else:
                    Layers.append(self.general_conv2d(Layers[-1], num_filters/(2**i), 3, stride=1, padding='SAME', activation_function='leakyrelu', do_norm=True, name=name + '_conv2d_' + str(i)))

            Layers.append(self.general_conv2d(Layers[-1], 2, 1, stride=1, padding='SAME', activation_function='None', do_norm=False, name=name + '_conv2d_' + str(i + 1)))
            Layers.append(tf.nn.softmax(Layers[-1], name=name + '_softmax'))

            return Layers

    def general_conv2d(self, input_data, filters=64,  kernel_size=7, stride=1, stddev=0.02, activation_function="relu", padding="VALID", do_norm=True, relu_factor=0, name="conv2d"):
        with tf.variable_scope(name):
            conv = tf.layers.conv2d(
                input_data, filters, kernel_size, stride, padding, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer())

            if do_norm:
                conv = tf.layers.batch_normalization(conv, momentum=0.9)

            if activation_function == "relu":
                conv = tf.nn.relu(conv, name='relu')
            if activation_function == "leakyrelu":
                conv = tf.nn.leaky_relu(conv, alpha=relu_factor)
            if activation_function == "elu":
                conv = tf.nn.elu(conv, name='elu')

            return conv

    def general_dense(self, input_data, units=1024, activation_function="relu", use_bias=True, kernel_initializer=None, bias_initializer=tf.zeros_initializer(), kernel_regularizer=None,bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, trainable=True, name='dense'):
        with tf.variable_scope(name):
            dense = tf.layers.dense(input_data, units, activation=None)

            if activation_function == "relu":
                dense = tf.nn.relu(dense, name='relu')
            if activation_function == "leakyrelu":
                dense = tf.nn.leaky_relu(dense, alpha=relu_factor)
            if activation_function == "elu":
                dense = tf.nn.elu(dense, name='elu')

            return dense
