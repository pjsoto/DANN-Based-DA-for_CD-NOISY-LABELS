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
import neptune.new as neptune
#from tensordash.tensordash import Tensordash, Customdash

from Tools import *
from Models_UDAB2 import *
from Amazonia_Legal_RO import AMAZON_RO
from Amazonia_Legal_PA import AMAZON_PA
from Cerrado_Biome_MA import CERRADO_MA
parser = argparse.ArgumentParser(description='')
#Defining the meta-paramerts
parser.add_argument('--method_type', dest='method_type', type=str, default='EF_CNN', help='method that will be used')
# Training parameters
parser.add_argument('--epochs', dest='epochs', type=int, default=100, help='number of epochs')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=32, help='number images in batch')
# Optimizer hyperparameters
parser.add_argument('--lr', dest='lr', type=float, default=0.01, help='initial learning rate for the optimizer')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.9, help='momentum term')
# Image_processing hyperparameters
parser.add_argument('--data_augmentation', dest='data_augmentation', type=eval, choices=[True, False], default=True, help='if data argumentation is applied to the data')
parser.add_argument('--source_vertical_blocks', dest='source_vertical_blocks', type=int, default=10, help='number of blocks which will divide the image vertically')
parser.add_argument('--source_horizontal_blocks', dest='source_horizontal_blocks', type=int, default=10, help='number of blocks which will divide the image horizontally')
parser.add_argument('--target_vertical_blocks', dest='target_vertical_blocks', type=int, default=10, help='number of blocks which will divide the image vertically')
parser.add_argument('--target_horizontal_blocks', dest='target_horizontal_blocks', type=int, default=10, help='number of blocks which will divide the image horizontally')
parser.add_argument('--fixed_tiles', dest='fixed_tiles', type=eval, choices=[True, False], default=True, help='decide if tiles will be choosen randomly or not')
parser.add_argument('--defined_before', dest='defined_before', type=eval, choices=[True, False], default=False, help='decide if tiles have been defined before in another instance')
parser.add_argument('--image_channels', dest='image_channels', type=int, default=7, help='number of image channels')
parser.add_argument('--patches_dimension', dest='patches_dimension', type=int, default=29, help= 'dimension of the extracted patches')
parser.add_argument('--stride_s', dest='stride_s', type=int, default= 3, help= 'stride step')
parser.add_argument('--stride_t', dest='stride_t', type=int, default= 3, help= 'stride step')
parser.add_argument('--compute_ndvi', dest='compute_ndvi', type=eval, choices=[True, False], default=True, help='Cumpute and stack the ndvi index to the rest of bands')
parser.add_argument('--balanced_tr', dest='balanced_tr', type=eval, choices=[True, False], default=True, help='Decide wether a balanced training will be performed')
parser.add_argument('--balanced_vl', dest='balanced_vl', type=eval, choices=[True, False], default=True, help='Decide wether a balanced in validation set during training will be performed')
parser.add_argument('--buffer', dest='buffer', type=eval, choices=[True, False], default=True, help='Decide wether a buffer around deforestated regions will be performed')
parser.add_argument('--source_buffer_dimension_out', dest='source_buffer_dimension_out', type=int, default=4, help='Dimension of the buffer outside of the area')
parser.add_argument('--source_buffer_dimension_in', dest='source_buffer_dimension_in', type=int, default=2, help='Dimension of the buffer inside of the area')
parser.add_argument('--target_buffer_dimension_out', dest='target_buffer_dimension_out', type=int, default=2, help='Dimension of the buffer outside of the area')
parser.add_argument('--target_buffer_dimension_in', dest='target_buffer_dimension_in', type=int, default=0, help='Dimension of the buffer inside of the area')
parser.add_argument('--porcent_of_last_reference_in_actual_reference', dest='porcent_of_last_reference_in_actual_reference', type=int, default=100, help='Porcent of number of pixels of last reference in the actual reference')
parser.add_argument('--num_classes', dest='num_classes', type=int, default=2, help='Number of classes comprised in both domains')
# Phase
parser.add_argument('--phase', dest='phase', default='train', help='train|test|compute_metrics')
parser.add_argument('--training_type', dest='training_type', type=str, default='domain_adaptation', help='classification|domain_adaptation')
parser.add_argument('--da_type', dest='da_type', type=str, default='CL', help='CL|DR|CL_DR')
parser.add_argument('--runs', dest='runs', type=int, default=1, help='number of executions of the algorithm')
# Early stop parameter
parser.add_argument('--patience', dest='patience', type=int, default=10, help='number of epochs without improvement to apply early stop')
#Checkpoint dir
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='', help='Domain adaptation checkpoints')
# Images dir and names
parser.add_argument('--source_dataset', dest='source_dataset', type=str, default='Amazon',help='The name of the dataset used in source domain')
parser.add_argument('--target_dataset', dest='target_dataset', type=str, default='Cerrado',help='The name of the dataset used in target domain')
parser.add_argument('--images_section', dest='images_section', type=str, default='Organized/Images/', help='Folder for the images')
parser.add_argument('--reference_section', dest='reference_section', type=str, default='Organized/References/', help='Folder for the reference')
parser.add_argument('--data_type', dest='data_type', type=str, default='.npy', help= 'Type of the input images and references')

parser.add_argument('--source_data_t1_year', dest='source_data_t1_year', type=str, default='', help='Year of the image taken at time 1 in source domain')
parser.add_argument('--source_data_t2_year', dest='source_data_t2_year', type=str, default='', help='Year of the image taken at time 2 in source domain')
parser.add_argument('--target_data_t1_year', dest='target_data_t1_year', type=str, default='', help='Year of the image taken at time 1 in target domain')
parser.add_argument('--target_data_t2_year', dest='target_data_t2_year', type=str, default='', help='Year of the image taken at time 2 in target domain')
parser.add_argument('--source_data_t1_name', dest='source_data_t1_name', type=str, default='', help='source image 1 name')
parser.add_argument('--source_data_t2_name', dest='source_data_t2_name', type=str, default='', help='source image 2 name')
parser.add_argument('--target_data_t1_name', dest='target_data_t1_name', type=str, default='', help='target image 1 name')
parser.add_argument('--target_data_t2_name', dest='target_data_t2_name', type=str, default='', help='target image 2 name')
parser.add_argument('--source_reference_t1_name', dest='source_reference_t1_name', type=str, default='', help='source reference 1 name')
parser.add_argument('--source_reference_t2_name', dest='source_reference_t2_name', type=str, default='', help='source reference 2 name')
parser.add_argument('--target_reference_t1_name', dest='target_reference_t1_name', type=str, default='', help='target reference 1 name')
parser.add_argument('--target_reference_t2_name', dest='target_reference_t2_name', type=str, default='', help='target reference 2 name')
#Architecture configuration
parser.add_argument('--FE_Architecture', dest='FE_Architecture', type=str, default='', help='Decide the architecture of the Feature Extractor(FE)')
#Dataset Main paths
parser.add_argument('--dataset_main_path', dest='dataset_main_path', type=str, default='/media/lvc/Dados/PEDROWORK/Trabajo_Domain_Adaptation/Dataset/', help='Dataset main path')
parser.add_argument('--checkpoint_results_main_path', dest='checkpoint_results_main_path', type=str, default='E:/PEDROWORK/Trabajo_Domain_Adaptation/Code/checkpoints_results/')


args = parser.parse_args()

def main():
    run = neptune.init(
    project="pjsotove/Domain-Adaptation-For-Change-Detection",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJhMjI4NTlkMS0zNzE4LTRjYTEtYWMwMi02MzQzMTY3ZWI5NzUifQ==",
    )  # your credentials
    print(args)

    if not os.path.exists(args.checkpoint_results_main_path + 'CHECKPOINTS/'):
        os.makedirs(args.checkpoint_results_main_path + 'CHECKPOINTS/')

    args.checkpoint_dir = args.checkpoint_results_main_path + 'CHECKPOINTS/' + args.checkpoint_dir

    if args.source_dataset == 'Amazon_RO':
        args.dataset = 'Amazonia_Legal/'
        args.buffer_dimension_in = args.source_buffer_dimension_in
        args.buffer_dimension_out = args.source_buffer_dimension_out
        args.data_t1_name = args.source_data_t1_name
        args.data_t2_name = args.source_data_t2_name
        args.reference_t1_name = args.source_reference_t1_name
        args.reference_t2_name = args.source_reference_t2_name
        dataset_s = AMAZON_RO(args)

    if args.source_dataset == 'Amazon_PA':
        args.dataset = 'Amazonia_Legal/'
        args.buffer_dimension_in = args.source_buffer_dimension_in
        args.buffer_dimension_out = args.source_buffer_dimension_out
        args.data_t1_name = args.source_data_t1_name
        args.data_t2_name = args.source_data_t2_name
        args.reference_t1_name = args.source_reference_t1_name
        args.reference_t2_name = args.source_reference_t2_name
        dataset_s = AMAZON_PA(args)

    if args.source_dataset == 'Cerrado_MA':
        args.dataset = 'Cerrado_Biome/'
        args.buffer_dimension_in = args.source_buffer_dimension_in
        args.buffer_dimension_out = args.source_buffer_dimension_out
        args.data_t1_name = args.source_data_t1_name
        args.data_t2_name = args.source_data_t2_name
        args.reference_t1_name = args.source_reference_t1_name
        args.reference_t2_name = args.source_reference_t2_name
        dataset_s = CERRADO_MA(args)

    if args.target_dataset == 'Amazon_RO':
        args.dataset = 'Amazonia_Legal/'
        args.buffer_dimension_in = args.target_buffer_dimension_in
        args.buffer_dimension_out = args.target_buffer_dimension_out
        args.data_t1_name = args.target_data_t1_name
        args.data_t2_name = args.target_data_t2_name
        args.reference_t1_name = args.target_reference_t1_name
        args.reference_t2_name = args.target_reference_t2_name
        dataset_t = AMAZON_RO(args)

    if args.target_dataset == 'Amazon_PA':
        args.dataset = 'Amazonia_Legal/'
        args.buffer_dimension_in = args.target_buffer_dimension_in
        args.buffer_dimension_out = args.target_buffer_dimension_out
        args.data_t1_name = args.target_data_t1_name
        args.data_t2_name = args.target_data_t2_name
        args.reference_t1_name = args.target_reference_t1_name
        args.reference_t2_name = args.target_reference_t2_name
        dataset_t = AMAZON_PA(args)

    if args.target_dataset == 'Cerrado_MA':
        args.dataset = 'Cerrado_Biome/'
        args.buffer_dimension_in = args.target_buffer_dimension_in
        args.buffer_dimension_out = args.target_buffer_dimension_out
        args.data_t1_name = args.target_data_t1_name
        args.data_t2_name = args.target_data_t2_name
        args.reference_t1_name = args.target_reference_t1_name
        args.reference_t2_name = args.target_reference_t2_name
        dataset_t = CERRADO_MA(args)

    print(np.shape(dataset_s.images_norm))
    print(np.shape(dataset_t.images_norm))

    for i in range(args.runs):
        dataset = []
        now = datetime.now()
        dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
        print(dt_string)
        if args.training_type == 'classification':
            args.save_checkpoint_path = args.checkpoint_dir + '/' + args.method_type + '_' + dt_string + '/'

        if args.training_type == 'domain_adaptation':
            args.save_checkpoint_path = args.checkpoint_dir + '/' + 'Tr_M_' + dt_string + '/'
        if not os.path.exists(args.save_checkpoint_path):
            os.makedirs(args.save_checkpoint_path)
            #Writing the args into a file
        with open(args.save_checkpoint_path + 'commandline_args.txt', 'w') as f:
            json.dump(args.__dict__, f, indent=2)

        args.vertical_blocks = args.source_vertical_blocks
        args.horizontal_blocks = args.source_horizontal_blocks
        args.stride = args.stride_s
        dataset_s.Tiles_Configuration(args, i)
        dataset_s.Coordinates_Creator(args, i)
        args.vertical_blocks = args.target_vertical_blocks
        args.horizontal_blocks = args.target_horizontal_blocks
        args.stride = args.stride_t
        dataset_t.Tiles_Configuration(args, i)
        dataset_t.Coordinates_Creator(args, i)
        dataset.append(dataset_s)
        dataset.append(dataset_t)
        print('[*]Initializing the model...')
        model = Models(args, dataset, run)
        print('[*]Start the training of the model...')
        model.Train()

if __name__=='__main__':
    main()
