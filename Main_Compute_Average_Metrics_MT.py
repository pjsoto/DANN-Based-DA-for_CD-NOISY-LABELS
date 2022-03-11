import os
import sys
import json
import argparse
import numpy as np
import tensorflow as tf
import skimage.morphology
from datetime import datetime
import matplotlib.pyplot as plt
from skimage.morphology import square, disk
from sklearn.preprocessing import StandardScaler

from Tools import*
from Models_UDAB2 import*
from Amazonia_Legal_RO import AMAZON_RO
from Amazonia_Legal_PA import AMAZON_PA
from Cerrado_Biome_MA import CERRADO_MA
parser = argparse.ArgumentParser(description='')
#Defining the meta-paramerts

parser.add_argument('--method_type', dest='method_type', type=str, default='DANN', help='method that will be used')
parser.add_argument('--vertical_blocks', dest='vertical_blocks', type=int, default=10, help='number of blocks which will divide the image vertically')
parser.add_argument('--horizontal_blocks', dest='horizontal_blocks', type=int, default=10, help='number of blocks which will divide the image horizontally')
parser.add_argument('--patches_dimension', dest='patches_dimension', type=int, default=29, help= 'dimension of the extracted patches')
parser.add_argument('--fixed_tiles', dest='fixed_tiles', type=eval, choices=[True, False], default=True, help='decide if tiles will be choosen randomly or not')
parser.add_argument('--defined_before', dest='defined_before', type=eval, choices=[True, False], default=False, help='decide if tiles have been defined before in another instance')
parser.add_argument('--buffer', dest='buffer', type=eval, choices=[True, False], default=True, help='Decide wether a buffer around deforestated regions will be performed')
parser.add_argument('--buffer_dimension_out', dest='buffer_dimension_out', type=int, default=4, help='Dimension of the buffer outside of the area')
parser.add_argument('--buffer_dimension_in', dest='buffer_dimension_in', type=int, default=2, help='Dimension of the buffer inside of the area')
parser.add_argument('--eliminate_regions', dest='eliminate_regions', type=eval, choices=[True, False], default=True, help='Decide if small regions will be taken into account')
parser.add_argument('--area_avoided', dest='area_avoided', type=int, default=69, help='area threshold that will be avoided')
parser.add_argument('--Npoints', dest='Npoints', type=float, default=50, help='Number of thresholds used to compute the curves')
parser.add_argument('--compute_ndvi', dest='compute_ndvi', type=eval, choices=[True, False], default=True, help='Cumpute and stack the ndvi index to the rest of bands')
parser.add_argument('--phase', dest='phase', default='compute_metrics', help='train, test, compute_metrics')
parser.add_argument('--training_type', dest='training_type', type=str, default='classification', help='classification|domain_adaptation')
parser.add_argument('--save_result_text', dest='save_result_text', type=eval, choices=[True, False], default = True, help='decide if a text file results is saved')
#Checkpoint dir
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./DA_prove_1', help='Domain adaptation checkpoints')
#Results dir
parser.add_argument('--results_dir', dest='results_dir', type=str, default='./results_DA_prove_1/', help='results will be saved here')
# Images dir and names
parser.add_argument('--dataset', dest='dataset', type=str, default='Amazonia_Legal/',help='The name of the dataset used')
parser.add_argument('--images_section', dest='images_section', type=str, default='Organized/Images/', help='Folder for the images')
parser.add_argument('--reference_section', dest='reference_section', type=str, default='Organized/References/', help='Folder for the reference')
parser.add_argument('--data_type', dest='data_type', type=str, default='.npy', help= 'Type of the input images and references')
parser.add_argument('--data_t1_year', dest='data_t1_year', type=str, default='2016', help='Year of the time 1 image')
parser.add_argument('--data_t2_year', dest='data_t2_year', type=str, default='2017', help='Year of the time 2 image')
parser.add_argument('--data_t1_name', dest='data_t1_name', type=str, default='18_07_2016_image', help='image 1 name')
parser.add_argument('--data_t2_name', dest='data_t2_name', type=str, default='21_07_2017_image', help='image 2 name')
parser.add_argument('--reference_t1_name', dest='reference_t1_name', type=str, default='PAST_REFERENCE_FOR_2017_EPSG32620', help='reference 1 name')
parser.add_argument('--reference_t2_name', dest='reference_t2_name', type=str, default='REFERENCE_2017_EPSG32620', help='reference 2 name')
#Dataset Main paths
parser.add_argument('--dataset_main_path', dest='dataset_main_path', type=str, default='/media/lvc/Dados/PEDROWORK/Trabajo_Domain_Adaptation/Dataset/', help='Dataset main path')
parser.add_argument('--checkpoint_results_main_path', dest='checkpoint_results_main_path', type=str, default='E:/PEDROWORK/Trabajo_Domain_Adaptation/Code/checkpoints_results/')
args = parser.parse_args()

def Main():

    if args.dataset == 'Amazon_RO':
        args.dataset = 'Amazonia_Legal/'
        dataset = AMAZON_RO(args)

    if args.dataset == 'Amazon_PA':
        args.dataset = 'Amazonia_Legal/'
        dataset = AMAZON_PA(args)

    if args.dataset == 'Cerrado_MA':
        args.dataset = 'Cerrado_Biome/'
        dataset = CERRADO_MA(args)

    if not os.path.exists( args.checkpoint_results_main_path + 'RESULTS_AVG/'):
        os.makedirs(args.checkpoint_results_main_path + 'RESULTS_AVG/')

    args.average_results_dir = args.checkpoint_results_main_path + 'RESULTS_AVG/' + args.results_dir + '/'
    if not os.path.exists(args.average_results_dir):
        os.makedirs(args.average_results_dir)

    args.results_dir = args.checkpoint_results_main_path + 'RESULTS/' + args.results_dir + '/'
    args.checkpoint_dir = args.checkpoint_results_main_path + 'CHECKPOINTS/' + args.checkpoint_dir + '/'
    counter = 0
    files = os.listdir(args.results_dir)
    for i in range(0, len(files)):
        if files[i] == 'Results.txt':
            print('Results file')
        else:
            Heat_map_path = args.results_dir + files[i] + '/heat_map.npy'
            if os.path.exists(Heat_map_path):
                heat_map = np.load(Heat_map_path)
                counter += 1
                if i == 0:
                    HEAT_MAP = np.zeros_like(heat_map)

                HEAT_MAP += heat_map

    dataset.Tiles_Configuration(args, 0)
    Avg_heat_map = HEAT_MAP/counter
    args.file = 'Avg_Scores'
    args.results_dir = args.average_results_dir
    if not os.path.exists(args.results_dir + args.file + '/'):
        os.makedirs(args.results_dir + args.file + '/')

    if args.save_result_text:
        # Open a file in order to save the training history
        f = open(args.results_dir + "Results.txt","a")
        if counter == 0:
            ACCURACY_ = []
            FSCORE_ = []
            RECALL_ = []
            PRECISION_ = []
            ALERT_AREA_ = []

    ACCURACY, FSCORE, RECALL, PRECISION, CONFUSION_MATRIX, ALERT_AREA = Metrics_For_Test_M(Avg_heat_map,
                                                                                        dataset.references[0], dataset.references[1],
                                                                                        dataset.Train_tiles, dataset.Valid_tiles, dataset.Undesired_tiles,
                                                                                        args)

    if args.save_result_text:

        ACCURACY_.append(ACCURACY[0,0])
        FSCORE_.append(FSCORE[0,0])
        RECALL_.append(RECALL[0,0])
        PRECISION_.append(PRECISION[0,0])
        ALERT_AREA_.append(ALERT_AREA[0,0])

        f.write("Run: %d Accuracy: %.2f%% F1-Score: %.2f%% Recall: %.2f%% Precision: %.2f%% Area: %.2f%% File Name: %s\n" % (counter, ACCURACY, FSCORE, RECALL, PRECISION, ALERT_AREA, args.file))
        f.close()
        print(ACCURACY_)
    else:
        print('Coming up!')



    if args.save_result_text:
        f = open(args.results_dir + "Results.txt","a")
        ACCURACY_m = np.mean(ACCURACY_)
        FSCORE_m = np.mean(FSCORE_)
        RECALL_m = np.mean(RECALL_)
        PRECISION_m = np.mean(PRECISION_)
        ALERT_AREA_m = np.mean(ALERT_AREA_)


        ACCURACY_s = np.std(ACCURACY_)
        FSCORE_s = np.std(FSCORE_)
        RECALL_s = np.std(RECALL_)
        PRECISION_s = np.std(PRECISION_)
        ALERT_AREA_s = np.std(ALERT_AREA_)

        f.write("Mean: %d Accuracy: %f%% F1-Score: %f%% Recall: %f%% Precision: %f%% Area: %f%%\n" % ( 0, ACCURACY_m, FSCORE_m, RECALL_m, PRECISION_m, ALERT_AREA_m))
        f.write("Std: %d Accuracy: %.2f%% F1-Score: %.2f%% Recall: %.2f%% Precision: %.2f%% Area: %.2f%%\n" % ( 0, ACCURACY_s, FSCORE_s, RECALL_s, PRECISION_s, ALERT_AREA_s))
        f.close()

if __name__=='__main__':
    Main()
