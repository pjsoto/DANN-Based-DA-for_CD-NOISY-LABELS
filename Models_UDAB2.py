import os
import sys
import time
import skimage
import numpy as np
import scipy.io as sio
from tqdm import trange
import tensorflow as tf
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from contextlib import redirect_stdout



from Tools import *
from Networks import *
from flip_gradient import flip_gradient
class Models():
    def __init__(self, args, dataset, run):
        self.args = args
        self.run = run
        # Initializing the placeholders
        #Changing  the seed  at any run
        tf.set_random_seed(int(time.time()))
        tf.reset_default_graph()
        self.learning_rate = tf.placeholder(tf.float32, [], name="learning_rate")
        if self.args.compute_ndvi:
            self.data = tf.placeholder(tf.float32, [None, self.args.patches_dimension, self.args.patches_dimension, 2 * self.args.image_channels + 2], name = "data")
        else:
            self.data = tf.placeholder(tf.float32, [None, self.args.patches_dimension, self.args.patches_dimension, 2 * self.args.image_channels], name = "data")

        self.label = tf.placeholder(tf.float32, [None, self.args.num_classes], name = "label")
        self.label_d = tf.placeholder(tf.float32, [None, 2], name = "label_d")
        self.mask_c = tf.placeholder(tf.float32, [None,], name="labeled_samples")
        self.L = tf.placeholder(tf.float32, [], name="L" )

        # Initializing the network class
        self.classifier = EF_CNN(self.args)
        # Initializing the models individually
        if self.args.FE_Architecture == 'Mabel_Arch':
            Encoder_Outputs = self.classifier.build_Mabel_Arch(self.data, reuse = False, name = "FE")
        elif self.args.FE_Architecture == 'Ganin_Arch':
            Encoder_Outputs = self.classifier.build_Ganin_Arch(self.data, reuse = False, name = "FE")
        #Defining the classifier
        Classifier_Outputs = self.classifier.build_MLP_1hidden_cl(Encoder_Outputs[-1], reuse = False, name = "MLP_Cl")

        self.logits_c = Classifier_Outputs[-2]
        self.prediction_c = Classifier_Outputs[-1]
        self.features_c = Encoder_Outputs[-1]

        if self.args.training_type == 'domain_adaptation':
            if 'DR' in self.args.da_type:
                flip_feature = flip_gradient(self.features_c, self.L)
                self.DR = Domain_Regressors(self.args)
                DR_Ouputs = self.DR.build_Domain_Classifier_Arch(flip_feature, name = 'FC_Domain_Classifier')
                self.logits_d = DR_Ouputs[-2]

        if self.args.phase == 'train':
            self.summary(Encoder_Outputs, 'Encoder:')
            self.summary(Classifier_Outputs, 'Classifier:')

            self.dataset_s = dataset[0]
            self.dataset_t = dataset[1]
            #Defining losses
            temp = tf.nn.softmax_cross_entropy_with_logits(logits = self.logits_c, labels = self.label)
            self.classifier_loss =  tf.reduce_sum(self.mask_c * temp) / tf.reduce_sum(self.mask_c)
            if self.args.training_type == 'classification':
                self.total_loss = self.classifier_loss
            else:
                if 'DR' in self.args.da_type:
                    self.summary(DR_Ouputs, "Domain_Regressor: ")
                    self.domainregressor_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.logits_d, labels = self.label_d))
                    self.total_loss = self.classifier_loss + self.domainregressor_loss
                else:
                    self.total_loss = self.classifier_loss
            #Defining the Optimizers
            self.training_optimizer = tf.train.MomentumOptimizer(self.learning_rate, self.args.beta1).minimize(self.total_loss)
            self.saver = tf.train.Saver(max_to_keep=5)
            self.sess=tf.Session()
            self.sess.run(tf.initialize_all_variables())

        elif self.args.phase == 'test':
            self.dataset = dataset
            self.saver = tf.train.Saver(max_to_keep=5)
            self.sess=tf.Session()
            self.sess.run(tf.initialize_all_variables())
            print('[*]Loading the feature extractor and classifier trained models...')
            mod = self.load(self.args.trained_model_path)
            if mod:
                print(" [*] Load with SUCCESS")
            else:
                print(" [!] Load failed...")
                sys.exit()

    def Learning_rate_decay(self):
        lr = self.args.lr / (1. + 10 * self.p)**0.75
        return lr

    def summary(self, net, name):
        print(net)
        f = open(self.args.save_checkpoint_path + "Architecture.txt","a")
        f.write(name + "\n")
        for i in range(len(net)):
            print(net[i].get_shape().as_list())
            f.write(str(net[i].get_shape().as_list()) + "\n")
        f.close()

    def Train(self):

        pat = 0
        f1score_val_cl = 0
        if self.args.balanced_tr:
            # Shuffling the data and labels
            central_pixels_coor_tr = self.dataset_s.central_pixels_coor_tr.copy()
            y_train = self.dataset_s.y_train.copy()
            central_pixels_coor_tr, y_train = shuffle(central_pixels_coor_tr, y_train, random_state=0)
            positive_coordinates = np.transpose(np.array(np.where(y_train == 1)))
            negative_coordinates = np.transpose(np.array(np.where(y_train == 0)))
            positive_coordinates = positive_coordinates[:,0]
            negative_coordinates = negative_coordinates[:,0]

            positive_central_pixels_coor_tr = central_pixels_coor_tr[positive_coordinates, :]
            if self.args.data_augmentation:
                positive_central_pixels_coor_tr, _ = Data_Augmentation_Definition(positive_central_pixels_coor_tr, np.ones((len(positive_coordinates),1)))

            #Taking the same amount of negative samples as positive
            negative_coordinates = negative_coordinates[:positive_central_pixels_coor_tr.shape[0]]
            if self.args.data_augmentation:
                negative_central_pixels_coor_tr = np.concatenate((central_pixels_coor_tr[negative_coordinates, :], np.zeros((len(negative_coordinates),1))),axis=1)
            else:
                negative_central_pixels_coor_tr = central_pixels_coor_tr[negative_coordinates, :]

            positive_y_train = np.ones((positive_central_pixels_coor_tr.shape[0],1))
            negative_y_train = np.zeros((negative_central_pixels_coor_tr.shape[0],1))
            central_pixels_coor_tr = np.concatenate((positive_central_pixels_coor_tr, negative_central_pixels_coor_tr), axis=0)
            y_train = np.concatenate((positive_y_train, negative_y_train), axis=0)
            # Shuffling again
            central_pixels_coor_tr_s, y_train_s = shuffle(central_pixels_coor_tr, y_train,random_state=0)
        if self.args.balanced_vl:
            central_pixels_coor_vl = self.dataset_s.central_pixels_coor_vl.copy()
            y_valid = self.dataset_s.y_valid.copy()
            # Shuffling the data and labels
            central_pixels_coor_vl, y_valid = shuffle(central_pixels_coor_vl, y_valid,random_state=0)

            positive_coordinates = np.transpose(np.array(np.where(y_valid == 1)))
            negative_coordinates = np.transpose(np.array(np.where(y_valid == 0)))
            positive_coordinates = positive_coordinates[:,0]
            negative_coordinates = negative_coordinates[:,0]

            positive_central_pixels_coor_vl = central_pixels_coor_vl[positive_coordinates, :]
            if self.args.data_augmentation:
                positive_central_pixels_coor_vl, _ = Data_Augmentation_Definition(positive_central_pixels_coor_vl, np.ones((len(positive_coordinates),1)))

            #Taking the same amount of negative samples as positive
            negative_coordinates = negative_coordinates[:positive_central_pixels_coor_vl.shape[0]]
            if self.args.data_augmentation:
                negative_central_pixels_coor_vl = np.concatenate((central_pixels_coor_vl[negative_coordinates, :] , np.zeros((len(negative_coordinates),1))), axis=1)
            else:
                negative_central_pixels_coor_vl = central_pixels_coor_vl[negative_coordinates, :]

            positive_y_valid = np.ones((positive_central_pixels_coor_vl.shape[0],1))
            negative_y_valid = np.zeros((negative_central_pixels_coor_vl.shape[0],1))
            central_pixels_coor_vl = np.concatenate((positive_central_pixels_coor_vl, negative_central_pixels_coor_vl), axis=0)
            y_valid = np.concatenate((positive_y_valid, negative_y_valid), axis=0)
            # Shuffling again
            central_pixels_coor_vl_s, y_valid_s = shuffle(central_pixels_coor_vl, y_valid,random_state=0)

        print('Source sets dimensions')
        print(np.shape(central_pixels_coor_tr_s))
        print(np.shape(central_pixels_coor_vl_s))
        print(np.shape(y_train_s))
        print(np.shape(y_valid_s))

        if self.args.training_type == 'classification':
            print('Classification training on source domain')
            y_train_s_hot = tf.keras.utils.to_categorical(y_train_s, self.args.num_classes)
            y_valid_s_hot = tf.keras.utils.to_categorical(y_valid_s, self.args.num_classes)

            central_pixels_coor_tr = central_pixels_coor_tr_s.copy()
            central_pixels_coor_vl = central_pixels_coor_vl_s.copy()

            y_train_c_hot = y_train_s_hot.copy()
            y_valid_c_hot = y_valid_s_hot.copy()
            y_train_d_hot = np.ones((y_train_c_hot.shape[0], 2))
            y_valid_d_hot = np.ones((y_valid_c_hot.shape[0], 2))
            classification_mask_tr = np.ones((y_train_c_hot.shape[0], 1))
            classification_mask_vl = np.ones((y_valid_c_hot.shape[0], 1))

            domain_indexs_tr = np.zeros((y_train_c_hot.shape[0], 1))
            domain_indexs_vl = np.zeros((y_valid_c_hot.shape[0], 1))

        if self.args.training_type == 'domain_adaptation':
            print('Applying Domain Adaptation')
            # Analysing the target train dataset
            central_pixels_coor_tr = self.dataset_t.central_pixels_coor_tr.copy()
            y_train = self.dataset_t.y_train.copy()
            central_pixels_coor_tr, y_train = shuffle(central_pixels_coor_tr, y_train, random_state=0)
            positive_coordinates = np.transpose(np.array(np.where(y_train == 1)))
            negative_coordinates = np.transpose(np.array(np.where(y_train == 0)))
            positive_coordinates = positive_coordinates[:,0]
            negative_coordinates = negative_coordinates[:,0]
            if len(negative_coordinates) != 0:
                positive_central_pixels_coor_tr = central_pixels_coor_tr[positive_coordinates, :]
                if self.args.data_augmentation:
                    positive_central_pixels_coor_tr, _ = Data_Augmentation_Definition(positive_central_pixels_coor_tr, np.ones((len(positive_coordinates),1)))

                #Taking the same amount of negative samples as positive
                negative_coordinates = negative_coordinates[:positive_central_pixels_coor_tr.shape[0]]
                if self.args.data_augmentation:
                    negative_central_pixels_coor_tr = np.concatenate((central_pixels_coor_tr[negative_coordinates, :], np.zeros((len(negative_coordinates),1))),axis=1)
                else:
                    negative_central_pixels_coor_tr = central_pixels_coor_tr[negative_coordinates, :]

                positive_y_train = np.ones((positive_central_pixels_coor_tr.shape[0],1))
                negative_y_train = np.zeros((negative_central_pixels_coor_tr.shape[0],1))
                central_pixels_coor_tr = np.concatenate((positive_central_pixels_coor_tr, negative_central_pixels_coor_tr), axis=0)
                y_train = np.concatenate((positive_y_train, negative_y_train), axis=0)
                # Shuffling again
                central_pixels_coor_tr_t, y_train_t = shuffle(central_pixels_coor_tr, y_train,random_state=0)
            else:
                positive_central_pixels_coor_tr = central_pixels_coor_tr[positive_coordinates, :]
                if self.args.data_augmentation:
                    positive_central_pixels_coor_tr, _ = Data_Augmentation_Definition(positive_central_pixels_coor_tr, np.ones((len(positive_coordinates),1)))

                central_pixels_coor_tr_t = positive_central_pixels_coor_tr.copy()
                y_train_t = np.ones((positive_central_pixels_coor_tr.shape[0],1))

            # Analysing the target valid dataset
            central_pixels_coor_vl = self.dataset_t.central_pixels_coor_vl.copy()
            y_valid = self.dataset_t.y_valid.copy()
            central_pixels_coor_vl, y_valid = shuffle(central_pixels_coor_vl, y_valid, random_state=0)
            positive_coordinates = np.transpose(np.array(np.where(y_valid == 1)))
            negative_coordinates = np.transpose(np.array(np.where(y_valid == 0)))
            positive_coordinates = positive_coordinates[:,0]
            negative_coordinates = negative_coordinates[:,0]
            if len(negative_coordinates) != 0:
                positive_central_pixels_coor_vl = central_pixels_coor_vl[positive_coordinates, :]
                if self.args.data_augmentation:
                    positive_central_pixels_coor_vl, _ = Data_Augmentation_Definition(positive_central_pixels_coor_vl, np.ones((len(positive_coordinates),1)))

                #Taking the same amount of negative samples as positive
                negative_coordinates = negative_coordinates[:positive_central_pixels_coor_vl.shape[0]]
                if self.args.data_augmentation:
                    negative_central_pixels_coor_vl = np.concatenate((central_pixels_coor_vl[negative_coordinates, :], np.zeros((len(negative_coordinates),1))),axis=1)
                else:
                    negative_central_pixels_coor_vl = central_pixels_coor_vl[negative_coordinates, :]

                positive_y_valid = np.ones((positive_central_pixels_coor_vl.shape[0],1))
                negative_y_valid = np.zeros((negative_central_pixels_coor_vl.shape[0],1))
                central_pixels_coor_vl = np.concatenate((positive_central_pixels_coor_vl, negative_central_pixels_coor_vl), axis=0)
                y_valid = np.concatenate((positive_y_valid, negative_y_valid), axis=0)
                # Shuffling again
                central_pixels_coor_vl_t, y_valid_t = shuffle(central_pixels_coor_vl, y_valid,random_state=0)
            else:
                positive_central_pixels_coor_vl = central_pixels_coor_vl[positive_coordinates, :]
                if self.args.data_augmentation:
                    positive_central_pixels_coor_vl, _ = Data_Augmentation_Definition(positive_central_pixels_coor_vl, np.ones((len(positive_coordinates),1)))
                y_valid_t = np.ones((positive_central_pixels_coor_vl.shape[0],1))
            print("Target sets dimensions")
            print(np.shape(central_pixels_coor_tr_t))
            print(np.shape(central_pixels_coor_vl_t))
            print(np.shape(y_train_t))
            print(np.shape(y_valid_t))

            #Verify the size of each set aiming at balancing both training sets
            size_s = np.size(y_train_s, 0)
            size_t = np.size(y_train_t, 0)
            if size_t > size_s:
                positive_coordinates = np.transpose(np.array(np.where(y_train_t == 1)))
                negative_coordinates = np.transpose(np.array(np.where(y_train_t == 0)))
                positive_coordinates = positive_coordinates[:,0]
                negative_coordinates = negative_coordinates[:,0]
                if len(negative_coordinates) != 0:
                    central_pixels_coor_tr_p = central_pixels_coor_tr_t[positive_coordinates,:]
                    central_pixels_coor_tr_n = central_pixels_coor_tr_t[negative_coordinates,:]
                    y_train_p = y_train_t[positive_coordinates,:]
                    y_train_n = y_train_t[negative_coordinates,:]
                    central_pixels_coor_tr_p = central_pixels_coor_tr_p[:int(size_s/2),:]
                    central_pixels_coor_tr_n = central_pixels_coor_tr_n[:int(size_s/2),:]
                    y_train_p = y_train_p[:int(size_s/2),:]
                    y_train_n = y_train_n[:int(size_s/2),:]
                    central_pixels_coor_tr = np.concatenate((central_pixels_coor_tr_p, central_pixels_coor_tr_n), axis=0)
                    y_train = np.concatenate((y_train_p, y_train_n), axis=0)
                    central_pixels_coor_tr_t, y_train_t = shuffle(central_pixels_coor_tr, y_train,random_state=0)
                else:
                    central_pixels_coor_tr_t = central_pixels_coor_tr_t[:size_s, :]
                    y_train_t = y_train_t[:size_s, :]
            elif size_s > size_t:
                positive_coordinates = np.transpose(np.array(np.where(y_train_s == 1)))
                negative_coordinates = np.transpose(np.array(np.where(y_train_s == 0)))
                positive_coordinates = positive_coordinates[:,0]
                negative_coordinates = negative_coordinates[:,0]

                central_pixels_coor_tr_p = central_pixels_coor_tr_s[positive_coordinates,:]
                central_pixels_coor_tr_n = central_pixels_coor_tr_s[negative_coordinates,:]
                y_train_p = y_train_t[positive_coordinates,:]
                y_train_n = y_train_t[negative_coordinates,:]
                central_pixels_coor_tr_p = central_pixels_coor_tr_p[:int(size_t/2),:]
                central_pixels_coor_tr_n = central_pixels_coor_tr_n[:int(size_t/2),:]
                y_train_p = y_train_p[:int(size_t/2),:]
                y_train_n = y_train_n[:int(size_t/2),:]
                central_pixels_coor_tr = np.concatenate((central_pixels_coor_tr_p, central_pixels_coor_tr_n), axis=0)
                y_train = np.concatenate((y_train_p, y_train_n), axis=0)
                central_pixels_coor_tr_s, y_train_s = shuffle(central_pixels_coor_tr, y_train,random_state=0)

            #Verify the size of each set aiming at balancing both validation sets
            size_s = np.size(y_valid_s, 0)
            size_t = np.size(y_valid_t, 0)
            if size_t > size_s:
                positive_coordinates = np.transpose(np.array(np.where(y_valid_t == 1)))
                negative_coordinates = np.transpose(np.array(np.where(y_valid_t == 0)))
                positive_coordinates = positive_coordinates[:,0]
                negative_coordinates = negative_coordinates[:,0]
                if len(negative_coordinates) != 0:
                    central_pixels_coor_vl_p = central_pixels_coor_vl_t[positive_coordinates,:]
                    central_pixels_coor_vl_n = central_pixels_coor_vl_t[negative_coordinates,:]
                    y_valid_p = y_valid_t[positive_coordinates,:]
                    y_valid_n = y_valid_t[negative_coordinates,:]
                    central_pixels_coor_vl_p = central_pixels_coor_vl_p[:int(size_s/2),:]
                    central_pixels_coor_vl_n = central_pixels_coor_vl_n[:int(size_s/2),:]
                    y_valid_p = y_valid_p[:int(size_s/2),:]
                    y_valid_n = y_valid_n[:int(size_s/2),:]
                    central_pixels_coor_vl = np.concatenate((central_pixels_coor_vl_p, central_pixels_coor_vl_n), axis=0)
                    y_valid = np.concatenate((y_valid_p, y_valid_n), axis=0)
                    central_pixels_coor_vl_t, y_valid_t = shuffle(central_pixels_coor_vl, y_valid, random_state=0)
                else:
                    central_pixels_coor_vl_t = central_pixels_coor_vl_t[:size_s, :]
                    y_valid_t = y_valid_t[:size_s, :]

            elif size_s > size_t:
                positive_coordinates = np.transpose(np.array(np.where(y_valid_s == 1)))
                negative_coordinates = np.transpose(np.array(np.where(y_valid_s == 0)))
                positive_coordinates = positive_coordinates[:,0]
                negative_coordinates = negative_coordinates[:,0]

                central_pixels_coor_vl_p = central_pixels_coor_vl_s[positive_coordinates,:]
                central_pixels_coor_vl_n = central_pixels_coor_vl_s[negative_coordinates,:]
                y_valid_p = y_valid_t[positive_coordinates,:]
                y_valid_n = y_valid_t[negative_coordinates,:]
                central_pixels_coor_vl_p = central_pixels_coor_vl_p[:int(size_t/2),:]
                central_pixels_coor_vl_n = central_pixels_coor_vl_n[:int(size_t/2),:]
                y_valid_p = y_valid_p[:int(size_t/2),:]
                y_valid_n = y_valid_n[:int(size_t/2),:]
                central_pixels_coor_vl = np.concatenate((central_pixels_coor_vl_p, central_pixels_coor_vl_n), axis=0)
                y_valid = np.concatenate((y_valid_p, y_valid_n), axis=0)
                central_pixels_coor_vl_s, y_train_s = shuffle(central_pixels_coor_vl, y_valid,random_state=0)

            print("Source and Target dimensions")
            print(np.shape(central_pixels_coor_tr_s))
            print(np.shape(central_pixels_coor_tr_t))
            print(np.shape(central_pixels_coor_vl_s))
            print(np.shape(central_pixels_coor_vl_t))

            #Preparing the sets for the training
            y_train_ds = np.zeros((y_train_s.shape[0], 1))
            y_valid_ds = np.zeros((y_train_s.shape[0], 1))
            y_train_dt = np.ones((y_train_t.shape[0], 1))
            y_valid_dt = np.ones((y_train_t.shape[0], 1))

            y_train_s_hot = tf.keras.utils.to_categorical(y_train_s, self.args.num_classes)
            y_valid_s_hot = tf.keras.utils.to_categorical(y_valid_s, self.args.num_classes)
            y_train_t_hot = tf.keras.utils.to_categorical(y_train_t, self.args.num_classes)
            y_valid_t_hot = tf.keras.utils.to_categorical(y_valid_t, self.args.num_classes)
            y_train_ds_hot = tf.keras.utils.to_categorical(y_train_ds, self.args.num_classes)
            y_valid_ds_hot = tf.keras.utils.to_categorical(y_valid_ds, self.args.num_classes)
            y_train_dt_hot = tf.keras.utils.to_categorical(y_train_dt, self.args.num_classes)
            y_valid_dt_hot = tf.keras.utils.to_categorical(y_valid_dt, self.args.num_classes)

            central_pixels_coor_tr = np.concatenate((central_pixels_coor_tr_s, central_pixels_coor_tr_t), axis = 0)
            central_pixels_coor_vl = np.concatenate((central_pixels_coor_vl_s, central_pixels_coor_vl_t), axis = 0)
            y_train_c_hot = np.concatenate((y_train_s_hot, y_train_t_hot), axis = 0)
            y_valid_c_hot = np.concatenate((y_valid_s_hot, y_valid_t_hot), axis = 0)
            y_train_d_hot = np.concatenate((y_train_ds_hot, y_train_dt_hot), axis = 0)
            y_valid_d_hot = np.concatenate((y_valid_ds_hot, y_valid_dt_hot), axis = 0)
            domain_indexs_tr = np.concatenate((y_train_ds, y_train_dt), axis = 0)
            domain_indexs_vl = np.concatenate((y_valid_ds, y_valid_dt), axis = 0)

            classification_mask_tr = np.concatenate((np.ones((y_train_ds.shape[0] , 1)), np.zeros((y_train_dt.shape[0] , 1))),axis = 0)
            classification_mask_vl = np.concatenate((np.ones((y_valid_ds.shape[0] , 1)), np.zeros((y_valid_dt.shape[0] , 1))),axis = 0)
            if 'CL' in self.args.da_type:
                classification_mask_tr = np.ones((domain_indexs_tr.shape[0] , 1))
                classification_mask_vl = np.ones((domain_indexs_vl.shape[0] , 1))

        data = []
        x_train_s = np.concatenate((self.dataset_s.images_norm[0], self.dataset_s.images_norm[1]), axis = 2)
        data.append(x_train_s)
        if self.args.training_type == 'domain_adaptation':
            x_train_t = np.concatenate((self.dataset_t.images_norm[0], self.dataset_t.images_norm[1]), axis = 2)
            data.append(x_train_t)

        #Computing the number of batches
        num_batches_tr = central_pixels_coor_tr.shape[0]//self.args.batch_size
        num_batches_vl = central_pixels_coor_vl.shape[0]//self.args.batch_size

        e = 0
        while (e < self.args.epochs):

            #Shuffling the data and the labels
            num_samples = central_pixels_coor_tr.shape[0]
            index = np.arange(num_samples)
            np.random.shuffle(index)
            central_pixels_coor_tr = central_pixels_coor_tr[index, :]
            y_train_c_hot = y_train_c_hot[index, :]
            y_train_d_hot = y_train_d_hot[index, :]
            classification_mask_tr = classification_mask_tr[index, :]

            domain_indexs_tr = domain_indexs_tr[index, :]

            num_samples = central_pixels_coor_vl.shape[0]
            index = np.arange(num_samples)
            np.random.shuffle(index)
            central_pixels_coor_vl = central_pixels_coor_vl[index, :]
            y_valid_c_hot = y_valid_c_hot[index, :]
            y_valid_d_hot = y_valid_d_hot[index, :]
            classification_mask_vl = classification_mask_vl[index, :]

            domain_indexs_vl = domain_indexs_vl[index, :]

            # Open a file in order to save the training history
            f = open(self.args.save_checkpoint_path + "Log.txt","a")
            #Initializing loss metrics
            loss_cl_tr = np.zeros((1 , 2))
            loss_cl_vl = np.zeros((1 , 2))
            loss_dr_tr = np.zeros((1 , 2))
            loss_dr_vl = np.zeros((1 , 2))

            accuracy_tr = 0
            f1_score_tr = 0
            recall_tr = 0
            precission_tr = 0

            accuracy_vl = 0
            f1_score_vl = 0
            recall_vl = 0
            precission_vl = 0

            #Computing some parameters
            self.p = float(e) / self.args.epochs
            self.l = 2. / (1. + np.exp(-10. * self.p)) - 1
            self.lr = self.Learning_rate_decay()
            print(self.p)
            print(self.lr)
            print(self.l)

            batch_counter_cl = 0
            batchs = trange(num_batches_tr)

            for b in batchs:
                central_pixels_coor_tr_batch = central_pixels_coor_tr[b * self.args.batch_size : (b + 1) * self.args.batch_size , :]
                domain_index_batch = domain_indexs_tr[b * self.args.batch_size : (b + 1) * self.args.batch_size, :]
                classification_mask_batch = classification_mask_tr[b * self.args.batch_size : (b + 1) * self.args.batch_size, :]

                if self.args.data_augmentation:
                    transformation_indexs_batch = central_pixels_coor_tr[b * self.args.batch_size : (b + 1) * self.args.batch_size , 2]

                y_train_c_hot_batch = y_train_c_hot[b * self.args.batch_size : (b + 1) * self.args.batch_size , :]
                y_train_d_hot_batch = y_train_d_hot[b * self.args.batch_size : (b + 1) * self.args.batch_size , :]
                #Extracting the data patches from it's coordinates
                data_batch = Patch_Extraction(data, central_pixels_coor_tr_batch, domain_index_batch, self.args.patches_dimension, True, 'reflect')
                # Perform data augmentation?
                if self.args.data_augmentation:
                    data_batch = Data_Augmentation_Execution(data_batch, transformation_indexs_batch)

                if self.args.training_type == 'classification':
                    _, batch_loss, batch_probs = self.sess.run([self.training_optimizer, self.classifier_loss, self.prediction_c],
                                                                    feed_dict={self.data: data_batch, self.label: y_train_c_hot_batch,
                                                                    self.mask_c: classification_mask_batch[:,0], self.learning_rate: self.lr})

                if self.args.training_type == 'domain_adaptation':
                    _, batch_loss, batch_probs, batch_loss_d = self.sess.run([self.training_optimizer, self.classifier_loss, self.prediction_c, self.domainregressor_loss],
                                                                                    feed_dict = {self.data: data_batch, self.label: y_train_c_hot_batch, self.label_d: y_train_d_hot_batch,
                                                                                                 self.mask_c: classification_mask_batch[:,0], self.L: self.l, self.learning_rate: self.lr})
                    loss_dr_tr[0 , 0] += batch_loss_d


                loss_cl_tr[0 , 0] += batch_loss
                y_train_predict_batch = np.argmax(batch_probs, axis = 1)
                y_train_batch = np.argmax(y_train_c_hot_batch, axis = 1)


                accuracy, f1score, recall, precission, conf_mat = compute_metrics(y_train_batch.astype(int), y_train_predict_batch.astype(int))

                accuracy_tr += accuracy
                f1_score_tr += f1score
                recall_tr += recall
                precission_tr += precission
                batch_counter_cl += 1


            loss_cl_tr = loss_cl_tr/batch_counter_cl
            accuracy_tr = accuracy_tr/batch_counter_cl
            f1_score_tr = f1_score_tr/batch_counter_cl
            recall_tr = recall_tr/batch_counter_cl
            precission_tr = precission_tr/batch_counter_cl
            print(batch_counter_cl)
            if self.args.training_type == 'domain_adaptation' and 'DR' in self.args.da_type:
                loss_dr_tr = loss_dr_tr/batch_counter_cl
                self.run["train/cl_loss"].log(loss_cl_tr[0 , 0])
                self.run["train/dr_loss"].log(loss_dr_tr[0 , 0])
                self.run["train/accuracy"].log(accuracy_tr)
                self.run["train/F1-Score"].log(f1_score_tr)
                print ("%d [Tr loss: %f, acc.: %.2f%%,  precission: %.2f%%, recall: %.2f%%, fscore: %.2f%%, Dr loss: %f]" % (e, loss_cl_tr[0 , 0], accuracy_tr, precission_tr, recall_tr, f1_score_tr, loss_dr_tr[0,0]))
                f.write("%d [Tr loss: %f, acc.: %.2f%%, precission: %.2f%%, recall: %.2f%%, fscore: %.2f%%, Dr loss: %f]\n" % (e, loss_cl_tr[0 , 0], accuracy_tr, precission_tr, recall_tr, f1_score_tr, loss_dr_tr[0,0]))
            else:
                self.run["train/cl_loss"].log(loss_cl_tr[0 , 0])
                self.run["train/accuracy"].log(accuracy_tr)
                self.run["train/F1-Score"].log(f1_score_tr)
                print ("%d [Tr loss: %f, acc.: %.2f%%, precission: %.2f%%, recall: %.2f%%, fscore: %.2f%%]" % (e, loss_cl_tr[0,0], accuracy_tr, precission_tr, recall_tr, f1_score_tr))
                f.write("%d [Tr loss: %f, acc.: %.2f%%, precission: %.2f%%, recall: %.2f%%, fscore: %.2f%%]\n" % (e, loss_cl_tr[0,0], accuracy_tr, precission_tr, recall_tr, f1_score_tr))

            #Computing the validation loss
            print('[*]Computing the validation loss...')
            batch_counter_cl = 0
            batchs = trange(num_batches_vl)

            for b in batchs:
                central_pixels_coor_vl_batch = central_pixels_coor_vl[b * self.args.batch_size : (b + 1) * self.args.batch_size , :]
                domain_index_batch = domain_indexs_vl[b * self.args.batch_size : (b + 1) * self.args.batch_size, :]
                classification_mask_batch = classification_mask_vl[b * self.args.batch_size : (b + 1) * self.args.batch_size, :]
                if self.args.data_augmentation:
                    transformation_indexs_batch = central_pixels_coor_vl[b * self.args.batch_size : (b + 1) * self.args.batch_size , 2]

                y_valid_c_hot_batch = y_valid_c_hot[b * self.args.batch_size : (b + 1) * self.args.batch_size , :]
                y_valid_d_hot_batch = y_valid_d_hot[b * self.args.batch_size : (b + 1) * self.args.batch_size , :]
                #Extracting the data patches from it's coordinates
                data_batch = Patch_Extraction(data, central_pixels_coor_vl_batch, domain_index_batch, self.args.patches_dimension, True, 'reflect')

                if self.args.data_augmentation:
                    data_batch = Data_Augmentation_Execution(data_batch, transformation_indexs_batch)

                if self.args.training_type == 'classification':
                    _, batch_loss, batch_probs = self.sess.run([self.training_optimizer, self.classifier_loss, self.prediction_c],
                                                                    feed_dict={self.data: data_batch, self.label: y_train_c_hot_batch,
                                                                    self.mask_c: classification_mask_batch[:,0], self.learning_rate: self.lr})

                if self.args.training_type == 'domain_adaptation':
                    _, batch_loss, batch_probs, batch_loss_d = self.sess.run([self.training_optimizer, self.classifier_loss, self.prediction_c, self.domainregressor_loss],
                                                                                    feed_dict = {self.data: data_batch, self.label: y_train_c_hot_batch, self.label_d: y_train_d_hot_batch,
                                                                                                 self.mask_c: classification_mask_batch[:,0], self.L: self.l, self.learning_rate: self.lr})
                    loss_dr_vl[0 , 0] += batch_loss_d

                loss_cl_vl[0 , 0] += batch_loss
                y_valid_batch = np.argmax(y_valid_c_hot_batch, axis = 1)
                y_valid_predict_batch = np.argmax(batch_probs, axis = 1)


                accuracy, f1score, recall, precission, conf_mat = compute_metrics(y_valid_batch.astype(int), y_valid_predict_batch.astype(int))

                accuracy_vl += accuracy
                f1_score_vl += f1score
                recall_vl += recall
                precission_vl += precission
                batch_counter_cl += 1

            loss_cl_vl = loss_cl_vl/(batch_counter_cl)
            accuracy_vl = accuracy_vl/(batch_counter_cl)
            f1_score_vl = f1_score_vl/(batch_counter_cl)
            recall_vl = recall_vl/(batch_counter_cl)
            precission_vl = precission_vl/(batch_counter_cl)

            if self.args.training_type == 'domain_adaptation' and 'DR' in self.args.da_type:
                loss_dr_vl = loss_dr_vl/batch_counter_cl
                self.run["valid/cl_loss"].log(loss_cl_vl[0 , 0])
                self.run["valid/dr_loss"].log(loss_dr_vl[0 , 0])
                self.run["valid/accuracy"].log(accuracy_vl)
                self.run["valid/F1-Score"].log(f1_score_vl)
                print ("%d [Vl loss: %f, acc.: %.2f%%,  precission: %.2f%%, recall: %.2f%%, fscore: %.2f%%, DrV loss: %f]" % (e, loss_cl_vl[0,0], accuracy_vl, precission_vl, recall_vl, f1_score_vl, loss_dr_vl[0 , 0]))
                f.write("%d [Vl loss: %f, acc.: %.2f%%, precission: %.2f%%, recall: %.2f%%, fscore: %.2f%%, DrV loss: %f]\n" % (e, loss_cl_vl[0,0], accuracy_vl, precission_vl, recall_vl, f1_score_vl, loss_dr_vl[0 , 0]))

            else:
                self.run["valid/cl_loss"].log(loss_cl_vl[0 , 0])
                self.run["valid/accuracy"].log(accuracy_vl)
                self.run["valid/F1-Score"].log(f1_score_vl)
                print ("%d [Vl loss: %f, acc.: %.2f%%,  precission: %.2f%%, recall: %.2f%%, fscore: %.2f%%]" % (e, loss_cl_vl[0,0], accuracy_vl, precission_vl, recall_vl, f1_score_vl))
                f.write("%d [Vl loss: %f, acc.: %.2f%%, precission: %.2f%%, recall: %.2f%%, fscore: %.2f%%]\n" % (e, loss_cl_vl[0,0], accuracy_vl, precission_vl, recall_vl, f1_score_vl))

            f.close()

            if f1score_val_cl < f1_score_vl:
                f1score_val_cl = f1_score_vl
                pat = 0

                print('[!]Saving best model ...')
                self.save(self.args.save_checkpoint_path, e)
            else:
                pat += 1
                if pat > self.args.patience:
                    break
            e += 1

    def Test(self):


        print(np.shape(self.dataset.images_norm[0]))
        print(np.shape(self.dataset.images_norm[1]))
        heat_map = np.zeros((self.dataset.images_norm[0].shape[0] + 2 * (self.args.patches_dimension//2), self.dataset.images_norm[0].shape[1] + 2 * (self.args.patches_dimension//2)))
        print(np.shape(heat_map))
        x_test = []
        data = np.concatenate((self.dataset.images_norm[0], self.dataset.images_norm[1]), axis = 2)
        x_test.append(data)

        num_batches_ts = self.dataset.central_pixels_coor_ts.shape[0]//self.args.batch_size
        batchs = trange(num_batches_ts)
        print(num_batches_ts)

        for b in batchs:
            self.central_pixels_coor_ts_batch = self.dataset.central_pixels_coor_ts[b * self.args.batch_size : (b + 1) * self.args.batch_size , :]
            self.x_test_batch = Patch_Extraction(x_test, self.central_pixels_coor_ts_batch, np.zeros((self.args.batch_size , 1)), self.args.patches_dimension, True, 'reflect')

            probs = self.sess.run(self.prediction_c,
                                         feed_dict={self.data: self.x_test_batch})

            for i in range(self.args.batch_size):
                heat_map[int(self.central_pixels_coor_ts_batch[i,0]), int(self.central_pixels_coor_ts_batch[i,1])] = probs[i,1]


        if (num_batches_ts * self.args.batch_size) < self.dataset.central_pixels_coor_ts.shape[0]:
            self.central_pixels_coor_ts_batch = self.dataset.central_pixels_coor_ts[num_batches_ts * self.args.batch_size : , :]
            self.x_test_batch = Patch_Extraction(x_test, self.central_pixels_coor_ts_batch, np.zeros((self.central_pixels_coor_ts_batch.shape[0] , 1)), self.args.patches_dimension, True, 'reflect')

            probs = self.sess.run(self.prediction_c,
                                         feed_dict={self.data: self.x_test_batch})
            for i in range(self.central_pixels_coor_ts_batch.shape[0]):
                heat_map[int(self.central_pixels_coor_ts_batch[i,0]), int(self.central_pixels_coor_ts_batch[i,1])] = probs[i,1]


        np.save(self.args.save_results_dir + 'heat_map', heat_map)

    def save(self, checkpoint_dir, epoch):
        model_name = "DANN.model"

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=epoch)
        print("Checkpoint Saved with SUCCESS!")

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")
        print(checkpoint_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            aux = 'model_example'
            for i in range(len(ckpt_name)):
                if ckpt_name[-i-1] == '-':
                    aux = ckpt_name[-i:]
                    break
            return aux
        else:
            return ''


def Metrics_For_Test(heat_map,
                     reference_t1, reference_t2,
                     Train_tiles, Valid_tiles, Undesired_tiles,
                     Thresholds,
                     args):

    half_dim = args.patches_dimension//2

    save_path = args.results_dir + args.file + '/'
    print('[*]Defining the initial central patches coordinates...')
    mask_init = mask_creation(reference_t1.shape[0], reference_t1.shape[1], args.horizontal_blocks, args.vertical_blocks, [], [], [])
    mask_final = mask_creation(reference_t1.shape[0], reference_t1.shape[1], args.horizontal_blocks, args.vertical_blocks, Train_tiles, Valid_tiles, Undesired_tiles)

    #mask_final = mask_final_.copy()
    mask_final[mask_final == 1] = 0
    mask_final[mask_final == 3] = 0
    mask_final[mask_final == 2] = 1
    heat_map = heat_map[half_dim : -half_dim, half_dim : -half_dim]
    Probs_init = heat_map
    positive_map_init = np.zeros_like(Probs_init)

    # Metrics containers
    ACCURACY = np.zeros((1, len(Thresholds)))
    FSCORE = np.zeros((1, len(Thresholds)))
    RECALL = np.zeros((1, len(Thresholds)))
    PRECISSION = np.zeros((1, len(Thresholds)))
    CONFUSION_MATRIX = np.zeros((2 , 2, len(Thresholds)))
    CLASSIFICATION_MAPS = np.zeros((len(Thresholds), heat_map.shape[0], heat_map.shape[1], 3))
    ALERT_AREA = np.zeros((1 , len(Thresholds)))


    print('[*]The metrics computation has started...')
    #Computing the metrics for each defined threshold
    for th in range(len(Thresholds)):
        print(Thresholds[th])

        positive_map_init = np.zeros_like(heat_map)
        reference_t1_copy = reference_t1.copy()

        threshold = Thresholds[th]
        positive_coordinates = np.transpose(np.array(np.where(Probs_init >= threshold)))
        positive_map_init[positive_coordinates[:,0].astype('int'), positive_coordinates[:,1].astype('int')] = 1

        if args.eliminate_regions:
            positive_map_init_ = skimage.morphology.area_opening(positive_map_init.astype('int'),area_threshold = args.area_avoided, connectivity=1)
            eliminated_samples = positive_map_init - positive_map_init_
        else:
            eliminated_samples = np.zeros_like(heat_map)


        reference_t1_copy = reference_t1_copy + eliminated_samples
        reference_t1_copy[reference_t1_copy == 2] = 1

        reference_t1_copy = reference_t1_copy - 1
        reference_t1_copy[reference_t1_copy == -1] = 1
        reference_t1_copy[reference_t2 == 2] = 0
        mask_f = mask_final * reference_t1_copy


        central_pixels_coordinates_ts_ = np.transpose(np.array(np.where(mask_f == 1)))
        y_test = reference_t2[central_pixels_coordinates_ts_[:,0].astype('int'), central_pixels_coordinates_ts_[:,1].astype('int')]

        Probs = heat_map[central_pixels_coordinates_ts_[:,0].astype('int'), central_pixels_coordinates_ts_[:,1].astype('int')]
        Probs[Probs >= Thresholds[th]] = 1
        Probs[Probs <  Thresholds[th]] = 0

        accuracy, f1score, recall, precission, conf_mat = compute_metrics(y_test.astype('int'), Probs.astype('int'))

        #Classification_map, _, _ = Classification_Maps(Probs, y_test, central_pixels_coordinates_ts_, hit_map)

        TP = conf_mat[1 , 1]
        FP = conf_mat[0 , 1]
        TN = conf_mat[0 , 0]
        FN = conf_mat[1 , 0]
        numerator = TP + FP

        denominator = TN + FN + FP + TP

        Alert_area = 100*(numerator/denominator)
        print(f1score)
        ACCURACY[0 , th] = accuracy
        FSCORE[0 , th] = f1score
        RECALL[0 , th] = recall
        PRECISSION[0 , th] = precission
        CONFUSION_MATRIX[: , : , th] = conf_mat
        #CLASSIFICATION_MAPS[th, :, :, :] = Classification_map
        ALERT_AREA[0 , th] = Alert_area

    #Saving the metrics as npy array
    if not args.save_result_text:
        np.save(save_path + 'Accuracy', ACCURACY)
        np.save(save_path + 'Fscore', FSCORE)
        np.save(save_path + 'Recall', RECALL)
        np.save(save_path + 'Precission', PRECISSION)
        np.save(save_path + 'Confusion_matrix', CONFUSION_MATRIX)
        #np.save(save_path + 'Classification_maps', CLASSIFICATION_MAPS)
        np.save(save_path + 'Alert_area', ALERT_AREA)

    print('Accuracy')
    print(ACCURACY)
    print('Fscore')
    print(FSCORE)
    print('Recall')
    print(RECALL)
    print('Precision')
    print(PRECISSION)
    print('Confusion matrix')
    print(CONFUSION_MATRIX[:,:,0])
    print('Alert_area')
    print(ALERT_AREA)

    return ACCURACY, FSCORE, RECALL, PRECISSION, CONFUSION_MATRIX, ALERT_AREA


def Metrics_For_Test_M(heat_map,
                     reference_t1, reference_t2,
                     Train_tiles, Valid_tiles, Undesired_tiles,
                     args):

    half_dim = args.patches_dimension//2

    save_path = args.results_dir + args.file + '/'
    print('[*]Defining the initial central patches coordinates...')
    mask_init = mask_creation(reference_t1.shape[0], reference_t1.shape[1], args.horizontal_blocks, args.vertical_blocks, [], [], [])
    mask_final = mask_creation(reference_t1.shape[0], reference_t1.shape[1], args.horizontal_blocks, args.vertical_blocks, Train_tiles, Valid_tiles, Undesired_tiles)

    #mask_final = mask_final_.copy()
    mask_final[mask_final == 1] = 0
    mask_final[mask_final == 3] = 0
    mask_final[mask_final == 2] = 1
    heat_map = heat_map[half_dim : -half_dim, half_dim : -half_dim]
    sio.savemat(save_path + 'heat_map.mat' , {'heat_map': heat_map})
    Probs_init = heat_map
    positive_map_init = np.zeros_like(Probs_init)

    reference_t1_copy_ = reference_t1.copy()
    reference_t1_copy_ = reference_t1_copy_ - 1
    reference_t1_copy_[reference_t1_copy_ == -1] = 1
    reference_t1_copy_[reference_t2 == 2] = 0
    mask_f_ = mask_final * reference_t1_copy_
    sio.savemat(save_path + 'mask_f_.mat' , {'mask_f_': mask_f_})
    sio.savemat(save_path + 'reference_t2.mat' , {'reference': reference_t2})
    # Raul Implementation
    min_array = np.zeros((1 , ))
    Pmax = np.max(Probs_init[mask_f_ == 1])
    probs_list = np.arange(Pmax, 0, -Pmax/(args.Npoints - 1))
    Thresholds = np.concatenate((probs_list , min_array))

    print('Max probability value:')
    print(Pmax)
    print('Thresholds:')
    print(Thresholds)
    # Metrics containers
    ACCURACY = np.zeros((1, len(Thresholds)))
    FSCORE = np.zeros((1, len(Thresholds)))
    RECALL = np.zeros((1, len(Thresholds)))
    PRECISSION = np.zeros((1, len(Thresholds)))
    CONFUSION_MATRIX = np.zeros((2 , 2, len(Thresholds)))
    CLASSIFICATION_MAPS = np.zeros((len(Thresholds), heat_map.shape[0], heat_map.shape[1], 3))
    ALERT_AREA = np.zeros((1 , len(Thresholds)))


    print('[*]The metrics computation has started...')
    #Computing the metrics for each defined threshold
    for th in range(len(Thresholds)):
        print(Thresholds[th])

        positive_map_init = np.zeros_like(heat_map)
        reference_t1_copy = reference_t1.copy()

        threshold = Thresholds[th]
        positive_coordinates = np.transpose(np.array(np.where(Probs_init >= threshold)))
        positive_map_init[positive_coordinates[:,0].astype('int'), positive_coordinates[:,1].astype('int')] = 1

        if args.eliminate_regions:
            positive_map_init_ = skimage.morphology.area_opening(positive_map_init.astype('int'),area_threshold = args.area_avoided, connectivity=1)
            eliminated_samples = positive_map_init - positive_map_init_
        else:
            eliminated_samples = np.zeros_like(heat_map)


        reference_t1_copy = reference_t1_copy + eliminated_samples
        reference_t1_copy[reference_t1_copy == 2] = 1
        reference_t1_copy = reference_t1_copy - 1
        reference_t1_copy[reference_t1_copy == -1] = 1
        reference_t1_copy[reference_t2 == 2] = 0
        mask_f = mask_final * reference_t1_copy


        central_pixels_coordinates_ts_ = np.transpose(np.array(np.where(mask_f == 1)))

        y_test = reference_t2[central_pixels_coordinates_ts_[:,0].astype('int'), central_pixels_coordinates_ts_[:,1].astype('int')]


        Probs = heat_map[central_pixels_coordinates_ts_[:,0].astype('int'), central_pixels_coordinates_ts_[:,1].astype('int')]
        Probs[Probs >= Thresholds[th]] = 1
        Probs[Probs <  Thresholds[th]] = 0

        accuracy, f1score, recall, precission, conf_mat = compute_metrics(y_test.astype('int'), Probs.astype('int'))

        #Classification_map, _, _ = Classification_Maps(Probs, y_test, central_pixels_coordinates_ts_, hit_map)

        TP = conf_mat[1 , 1]
        FP = conf_mat[0 , 1]
        TN = conf_mat[0 , 0]
        FN = conf_mat[1 , 0]
        numerator = TP + FP

        denominator = TN + FN + FP + TP

        Alert_area = 100*(numerator/denominator)
        #print(f1score)
        print(precission)
        print(recall)
        ACCURACY[0 , th] = accuracy
        FSCORE[0 , th] = f1score
        RECALL[0 , th] = recall
        PRECISSION[0 , th] = precission
        CONFUSION_MATRIX[: , : , th] = conf_mat
        #CLASSIFICATION_MAPS[th, :, :, :] = Classification_map
        ALERT_AREA[0 , th] = Alert_area

    #Saving the metrics as npy array
    if not args.save_result_text:
        np.save(save_path + 'Accuracy', ACCURACY)
        np.save(save_path + 'Fscore', FSCORE)
        np.save(save_path + 'Recall', RECALL)
        np.save(save_path + 'Precission', PRECISSION)
        np.save(save_path + 'Confusion_matrix', CONFUSION_MATRIX)
        #np.save(save_path + 'Classification_maps', CLASSIFICATION_MAPS)
        np.save(save_path + 'Alert_area', ALERT_AREA)

    print('Accuracy')
    print(ACCURACY)
    print('Fscore')
    print(FSCORE)
    print('Recall')
    print(RECALL)
    print('Precision')
    print(PRECISSION)
    print('Confusion matrix')
    print(CONFUSION_MATRIX[:,:,0])
    print('Alert_area')
    print(ALERT_AREA)

    return ACCURACY, FSCORE, RECALL, PRECISSION, CONFUSION_MATRIX, ALERT_AREA
