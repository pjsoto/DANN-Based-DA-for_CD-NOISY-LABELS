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


from Tools import *
from Models_UDAB2 import *
from Amazonia_Legal_RO import AMAZON_RO
from Amazonia_Legal_PA import AMAZON_PA
from Cerrado_Biome_MA import CERRADO_MA

parser = argparse.ArgumentParser(description='')
#Defining the meta-paramerts
parser.add_argument('--method_type', dest='method_type', type=str, default='EF_CNN', help='method that will be used')
# Testing parameters
parser.add_argument('--batch_size', dest='batch_size', type=int, default=4000, help='number images in batch')
parser.add_argument('--vertical_blocks', dest='vertical_blocks', type=int, default=10, help='number of blocks which will divide the image vertically')
parser.add_argument('--horizontal_blocks', dest='horizontal_blocks', type=int, default=10, help='number of blocks which will divide the image horizontally')
parser.add_argument('--image_channels', dest='image_channels', type=int, default=7, help='number of image channels')
parser.add_argument('--patches_dimension', dest='patches_dimension', type=int, default=29, help= 'dimension of the extracted patches')
parser.add_argument('--compute_ndvi', dest='compute_ndvi', type=eval, choices=[True, False], default=True, help='Cumpute and stack the ndvi index to the rest of bands')
parser.add_argument('--buffer', dest='buffer', type=eval, choices=[True, False], default=False, help='Decide wether a buffer around deforestated regions will be performed')
parser.add_argument('--num_classes', dest='num_classes', type=int, default=2, help='Number of classes comprised in the domain')
# Phase
parser.add_argument('--phase', dest='phase', default='test', help='train|test|compute_metrics')
parser.add_argument('--training_type', dest='training_type', type=str, default='classification', help='classification|domain_adaptation')
#Checkpoint dir
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./DA_prove', help='Domain adaptation checkpoints')
parser.add_argument('--results_dir', dest='results_dir', type=str, default='./results_DA_prove', help='results will be saved here')
# Images dir and names
parser.add_argument('--dataset', dest='dataset', type=str, default='Amazonia_Legal/',help='The name of the dataset used')
parser.add_argument('--images_section', dest='images_section', type=str, default='Organized/Images/', help='Folder for the images')
parser.add_argument('--reference_section', dest='reference_section', type=str, default='Organized/References/', help='Folder for the reference')
parser.add_argument('--data_type', dest='data_type', type=str, default='.npy', help= 'Type of the input images and references')
parser.add_argument('--data_t1_year', dest='data_t1_year', type=str, default='2016', help='Year of the image taken at time 1')
parser.add_argument('--data_t2_year', dest='data_t2_year', type=str, default='2017', help='Year of the image taken at time 2')
parser.add_argument('--data_t1_name', dest='data_t1_name', type=str, default='18_07_2016_image', help='image 1 name')
parser.add_argument('--data_t2_name', dest='data_t2_name', type=str, default='21_07_2017_image', help='image 2 name')
parser.add_argument('--reference_t1_name', dest='reference_t1_name', type=str, default='PAST_REFERENCE_FOR_2017_EPSG32620', help='reference 1 name')
parser.add_argument('--reference_t2_name', dest='reference_t2_name', type=str, default='REFERENCE_2017_EPSG32620', help='reference 2 name')
#Dataset Main paths
parser.add_argument('--dataset_main_path', dest='dataset_main_path', type=str, default='/media/lvc/Dados/PEDROWORK/Trabajo_Domain_Adaptation/Dataset/', help='Dataset main path')
#Architecture configuration
parser.add_argument('--FE_Architecture', dest='FE_Architecture', type=str, default='', help='Decide the architecture of the Feature Extractor(FE)')
parser.add_argument('--checkpoint_results_main_path', dest='checkpoint_results_main_path', type=str, default='E:/PEDROWORK/Trabajo_Domain_Adaptation/Code/checkpoints_results/')
args = parser.parse_args()

def main():

    if args.phase == 'test':
        print(args)
        if not os.path.exists(args.checkpoint_results_main_path + 'results/'):
            os.makedirs(args.checkpoint_results_main_path + 'results/')

        args.results_dir = args.checkpoint_results_main_path + 'results/' + args.results_dir + '/'
        args.checkpoint_dir = args.checkpoint_results_main_path + 'checkpoints/' + args.checkpoint_dir + '/'


        if args.dataset == 'Amazon_RO':
            args.dataset = 'Amazonia_Legal/'
            dataset = AMAZON_RO(args)

        if args.dataset == 'Amazon_PA':
            args.dataset = 'Amazonia_Legal/'
            dataset = AMAZON_PA(args)

        if args.dataset == 'Cerrado_MA':
            args.dataset = 'Cerrado_Biome/'
            dataset = CERRADO_MA(args)

        dataset.Tiles_Configuration(args, 0)
        dataset.Coordinates_Creator(args, 0)

        checkpoint_files = os.listdir(args.checkpoint_dir)
        for i in range(len(checkpoint_files)):

            model_folder = checkpoint_files[i]
            args.trained_model_path = args.checkpoint_dir + '/' + model_folder + '/'
            model_folder_fields = model_folder.split('_')

            now = datetime.now()
            dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
            args.save_results_dir = args.results_dir + args.method_type + '_' + 'Model_Results_' + 'Trained_' + model_folder_fields[3] + '_' + model_folder_fields[4] + '_' + model_folder[-19:] + '_Tested_' + args.data_t1_year + '_' + args.data_t2_year + '_' + dt_string +'/'
            if not os.path.exists(args.save_results_dir):
                os.makedirs(args.save_results_dir)

            print('[*]Initializing the model...')
            model = Models(args, dataset)
            print('[*]Starting the test...')
            model.Test()

if __name__=='__main__':
    main()
