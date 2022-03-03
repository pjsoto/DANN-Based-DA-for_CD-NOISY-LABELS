import os
import numpy as np
import scipy.io as sio
import skimage as sk
#from osgeo import gdal
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix


def save_as_mat(data, name):
    sio.savemat(name, {name: data})

def Read_TIFF_Image(Path):
    img =[]
    #gdal_header = gdal.Open(Path)
    #img = gdal_header.ReadAsArray()
    return img

def Compute_NDVI_Band(Image):
    Image = Image.astype(np.float32)
    nir_band = Image[4, :, :]
    red_band = Image[3, :, :]
    ndvi = np.zeros((Image.shape[1] , Image.shape[2] , 1))
    ndvi[ : , : , 0] = np.divide((nir_band-red_band),(nir_band+red_band))
    return ndvi

def compute_metrics(true_labels, predicted_labels):
    accuracy = 100*accuracy_score(true_labels, predicted_labels)
    f1score = 100*f1_score(true_labels, predicted_labels)
    recall = 100*recall_score(true_labels, predicted_labels)
    prescision = 100*precision_score(true_labels, predicted_labels)
    conf_mat = confusion_matrix(true_labels, predicted_labels)
    return accuracy, f1score, recall, prescision, conf_mat

def Data_Augmentation_Definition(central_pixels_coor, labels):
    num_sample = np.size(central_pixels_coor , 0)
    data_cols = np.size(central_pixels_coor , 1)
    num_classes = np.size(labels, 1)

    #central_pixels_coor_augmented = np.zeros((8 * num_sample, data_cols + 1))
    central_pixels_coor_augmented = np.zeros((3 * num_sample, data_cols + 1))
    #labels_augmented = np.zeros((8 * num_sample, num_classes))
    labels_augmented = np.zeros((3 * num_sample, num_classes))
    counter = 0
    for s in range(num_sample):
        central_pixels_coor_x_0 = central_pixels_coor[s, :]
        labels_y_0 = labels[s, :]

        central_pixels_coor_augmented[counter, 0 : 2] = central_pixels_coor_x_0
        central_pixels_coor_augmented[counter, 2] = 1
        labels_augmented[counter, :] = labels_y_0
        counter += 1

        central_pixels_coor_augmented[counter, 0 : 2] = central_pixels_coor_x_0
        central_pixels_coor_augmented[counter, 2] = 2
        labels_augmented[counter, :] = labels_y_0
        counter += 1

        central_pixels_coor_augmented[counter, 0 : 2] = central_pixels_coor_x_0
        central_pixels_coor_augmented[counter, 2] = 3
        labels_augmented[counter, :] = labels_y_0
        counter += 1

    return central_pixels_coor_augmented, labels_augmented

def Data_Augmentation_Execution(data, transformation_indexs):
    data_rows = np.size(data , 1)
    data_cols = np.size(data , 2)
    data_depth = np.size(data , 3)
    num_sample = np.size(data , 0)

    data_transformed = np.zeros((num_sample, data_rows, data_cols, data_depth))
    counter = 0
    for s in range(num_sample):
        data_x_0 = data[s, :, :, :]
        transformation_index = transformation_indexs[s]
        #Rotating
        if transformation_index == 0:
            data_transformed[s, :, :, :] = data_x_0
        if transformation_index == 1:
            data_transformed[s, :, :, :] = np.rot90(data_x_0)
        if transformation_index == 2:
            data_transformed[s, :, :, :] = np.flip(data_x_0, 0)
        if transformation_index == 3:
            data_transformed[s, :, :, :] = np.flip(data_x_0, 1)
    return data_transformed

def Patch_Extraction(data, central_pixels_indexs, domain_index, patch_size, padding, mode):

    half_dim = patch_size // 2
    data_rows = np.size(data[0], 0)
    data_cols = np.size(data[0], 1)
    data_depth = np.size(data[0], 2)
    num_samp = np.size(central_pixels_indexs , 0)

    patches_cointainer = np.zeros((num_samp, patch_size, patch_size, data_depth))

    if padding:
        data_padded = []
        for i in range(len(data)):
            if mode == 'zeros':
                upper_padding = np.zeros((half_dim, data_cols, data_depth))
                left_padding = np.zeros((data_rows + half_dim, half_dim, data_depth))
                bottom_padding = np.zeros((half_dim, half_dim + data_cols, data_depth))
                right_padding = np.zeros((2 * half_dim + data_rows, half_dim, data_depth))

                #Add padding to the data
                data_padded_ = np.concatenate((upper_padding, data[i]), axis=0)
                data_padded_ = np.concatenate((left_padding, data_padded_), axis=1)
                data_padded_ = np.concatenate((data_padded_, bottom_padding), axis=0)
                data_padded_ = np.concatenate((data_padded_, right_padding), axis=1)
                data_padded.append(data_padded_)
            if mode == 'reflect':
                npad = ((half_dim , half_dim) , (half_dim , half_dim) , (0 , 0))
                data_padded.append(np.pad(data[i], pad_width = npad, mode = 'reflect'))
    else:
        data_padded = data
    # ESto aqui tiene ser revisado para el nuevo contexto.
    for i in range(num_samp):
        data_padded_ = data_padded[int(domain_index[i,0])]
        patches_cointainer[i, :, :, :] = data_padded_[int(central_pixels_indexs[i , 0]) - half_dim  : int(central_pixels_indexs[i , 0]) + half_dim + 1, int(central_pixels_indexs[i , 1]) - half_dim : int(central_pixels_indexs[i , 1]) + half_dim + 1, :]

    return patches_cointainer

def mask_creation(mask_row, mask_col, num_patch_row, num_patch_col, Train_tiles, Valid_tiles, Undesired_tiles):
    train_index = 1
    teste_index = 2
    valid_index = 3
    undesired_index = 4

    patch_dim_row = mask_row//num_patch_row
    patch_dim_col = mask_col//num_patch_col

    mask_array = 2 * np.ones((mask_row, mask_col))

    train_mask = np.ones((patch_dim_row, patch_dim_col))
    valid_mask = 3 * np.ones((patch_dim_row, patch_dim_col))
    undesired_mask = 4 * np.ones((patch_dim_row, patch_dim_col))
    counter_r = 1
    counter = 1
    for i in range(0, mask_row, patch_dim_row):
        for j in range(0 , mask_col, patch_dim_col):
            train = np.size(np.where(Train_tiles == counter),1)
            valid = np.size(np.where(Valid_tiles == counter),1)
            undesired = np.size(np.where(Undesired_tiles == counter), 1)
            if train == 1:
                mask_array[i : i + patch_dim_row, j : j + patch_dim_col] = train_mask
                if counter_r == num_patch_row:
                    mask_array[i : mask_row, j : j + patch_dim_col] = np.ones((mask_row - i, patch_dim_col))
            if valid == 1:
                mask_array[i : i + patch_dim_row, j : j + patch_dim_col] = valid_mask
                if counter_r == num_patch_row:
                    mask_array[i : mask_row, j : j + patch_dim_col] = 3 * np.ones((mask_row - i, patch_dim_col))
            if undesired == 1:
                mask_array[i : i + patch_dim_row, j : j + patch_dim_col] = undesired_mask
                if counter_r == num_patch_row:
                    mask_array[i : mask_row, j : j + patch_dim_col] = 4 * np.ones((mask_row - i, patch_dim_col))

            counter += 1
        counter_r += 1
    return mask_array

def Central_Pixel_Definition(mask, last_reference, actual_reference, patch_dimension, stride, porcent_of_last_reference_in_actual_reference):

    mask_rows = np.size(mask, 0)
    mask_cols = np.size(mask, 1)

    half_dim = patch_dimension//2
    upper_padding = np.zeros((half_dim, mask_cols))
    left_padding = np.zeros((mask_rows + half_dim, half_dim))
    bottom_padding = np.zeros((half_dim, half_dim + mask_cols))
    right_padding = np.zeros((2 * half_dim + mask_rows, half_dim))

    #Add padding to the mask
    mask_padded = np.concatenate((upper_padding, mask), axis=0)
    mask_padded = np.concatenate((left_padding, mask_padded), axis=1)
    mask_padded = np.concatenate((mask_padded, bottom_padding), axis=0)
    mask_padded = np.concatenate((mask_padded, right_padding), axis=1)

    #Add padding to the last reference
    last_reference_padded = np.concatenate((upper_padding, last_reference), axis=0)
    last_reference_padded = np.concatenate((left_padding, last_reference_padded), axis=1)
    last_reference_padded = np.concatenate((last_reference_padded, bottom_padding), axis=0)
    last_reference_padded = np.concatenate((last_reference_padded, right_padding), axis=1)

    #Add padding to the last reference
    actual_reference_padded = np.concatenate((upper_padding, actual_reference), axis=0)
    actual_reference_padded = np.concatenate((left_padding, actual_reference_padded), axis=1)
    actual_reference_padded = np.concatenate((actual_reference_padded, bottom_padding), axis=0)
    actual_reference_padded = np.concatenate((actual_reference_padded, right_padding), axis=1)

    #Initializing the central pixels coordinates containers
    central_pixels_coord_tr_init = []
    central_pixels_coord_vl_init = []

    if stride == 1:
        central_pixels_coord_tr_init = np.where(mask_padded == 1)
        central_pixels_coord_vl_init = np.where(mask_padded == 3)
        central_pixels_coord_tr_init = np.transpose(np.array(central_pixels_coord_tr_init))
        central_pixels_coord_vl_init = np.transpose(np.array(central_pixels_coord_vl_init))
    else:
        counter_tr = 0
        counter_vl = 0
        for i in range(2 * half_dim, np.size(mask_padded , 0) - 2 * half_dim, stride):
            for j in range(2 * half_dim, np.size(mask_padded , 1) - 2 * half_dim, stride):
                mask_value = mask_padded[i , j]
                #print(mask_value)
                if mask_value == 1:
                    #Belongs to the training tile
                    counter_tr += 1

                if mask_value == 3:
                    #Belongs to the validation tile
                    counter_vl += 1

        central_pixels_coord_tr_init = np.zeros((counter_tr, 2))
        central_pixels_coord_vl_init = np.zeros((counter_vl, 2))
        counter_tr = 0
        counter_vl = 0
        for i in range(2 * half_dim , np.size(mask_padded , 0) - 2 * half_dim, stride):
            for j in range(2 * half_dim , np.size(mask_padded , 1) - 2 * half_dim, stride):
                mask_value = mask_padded[i , j]
                #print(mask_value)
                if mask_value == 1:
                    #Belongs to the training tile
                    central_pixels_coord_tr_init[counter_tr , 0] = int(i)
                    central_pixels_coord_tr_init[counter_tr , 1] = int(j)
                    counter_tr += 1
                if mask_value == 3:
                    #Belongs to the validation tile
                    central_pixels_coord_vl_init[counter_vl , 0] = int(i)
                    central_pixels_coord_vl_init[counter_vl , 1] = int(j)
                    counter_vl += 1


    #Refine the central pixels coordinates
    counter_tr = 0
    counter_vl = 0
    for i in range(np.size(central_pixels_coord_tr_init , 0)):
        coordinates = [int(central_pixels_coord_tr_init[i , 0]) - half_dim  , int(central_pixels_coord_tr_init[i , 0]) + half_dim + 1, int(central_pixels_coord_tr_init[i , 1]) - half_dim , int(central_pixels_coord_tr_init[i , 1]) + half_dim + 1]
        mask_reference_value = mask_padded[int(coordinates[0]) : int(coordinates[1]) , int(coordinates[2]) : int(coordinates[3])]
        last_reference_value = last_reference_padded[int(central_pixels_coord_tr_init[i, 0]) , int(central_pixels_coord_tr_init[i, 1])]
        actual_reference_value = actual_reference_padded[int(central_pixels_coord_tr_init[i, 0]) , int(central_pixels_coord_tr_init[i, 1])]
        test_pixels_indexs = np.transpose(np.array(np.where(mask_reference_value == 2)))
        if np.size(test_pixels_indexs,0) == 0:
            if (last_reference_value != 1) and (actual_reference_value <= 1):
                last_reference_patch = last_reference_padded[int(central_pixels_coord_tr_init[i, 0]) - half_dim : int(central_pixels_coord_tr_init[i, 0]) + half_dim + 1, int(central_pixels_coord_tr_init[i, 1]) - half_dim : int(central_pixels_coord_tr_init[i, 1]) + half_dim + 1]
                number_of_last_reference_pixels_indexs = np.where(last_reference_patch == 1)
                number_of_last_reference_pixels = np.size(number_of_last_reference_pixels_indexs , 1)
                porcent_of_last_reference = (number_of_last_reference_pixels/(patch_dimension*patch_dimension))*100
                if porcent_of_last_reference <= porcent_of_last_reference_in_actual_reference:
                    counter_tr += 1
    for i in range(np.size(central_pixels_coord_vl_init , 0)):
        coordinates = [int(central_pixels_coord_vl_init[i , 0]) - half_dim  , int(central_pixels_coord_vl_init[i , 0]) + half_dim + 1, int(central_pixels_coord_vl_init[i , 1]) - half_dim , int(central_pixels_coord_vl_init[i , 1]) + half_dim + 1]
        mask_reference_value = mask_padded[int(coordinates[0]) : int(coordinates[1]) , int(coordinates[2]) : int(coordinates[3])]
        last_reference_value = last_reference_padded[int(central_pixels_coord_vl_init[i, 0]) , int(central_pixels_coord_vl_init[i, 1])]
        actual_reference_value = actual_reference_padded[int(central_pixels_coord_vl_init[i, 0]) , int(central_pixels_coord_vl_init[i, 1])]
        test_pixels_indexs = np.transpose(np.array(np.where(mask_reference_value == 2)))
        if np.size(test_pixels_indexs,0) == 0:
            if (last_reference_value != 1 ) and (actual_reference_value <= 1):
                last_reference_patch = last_reference_padded[int(central_pixels_coord_vl_init[i, 0]) - half_dim : int(central_pixels_coord_vl_init[i, 0]) + half_dim  + 1, int(central_pixels_coord_vl_init[i, 1]) - half_dim : int(central_pixels_coord_vl_init[i, 1]) + half_dim + 1]
                number_of_last_reference_pixels_indexs = np.where(last_reference_patch == 1)
                number_of_last_reference_pixels = np.size(number_of_last_reference_pixels_indexs , 1)
                porcent_of_last_reference = (number_of_last_reference_pixels/(patch_dimension*patch_dimension))*100
                if porcent_of_last_reference <= porcent_of_last_reference_in_actual_reference:
                    counter_vl += 1

    central_pixels_coord_tr = np.zeros((counter_tr, 2))
    central_pixels_coord_vl = np.zeros((counter_vl, 2))
    y_train_init = np.zeros((counter_tr,1))
    y_valid_init = np.zeros((counter_vl,1))
    counter_tr = 0
    counter_vl = 0
    for i in range(np.size(central_pixels_coord_tr_init , 0)):
        coordinates = [int(central_pixels_coord_tr_init[i , 0]) - half_dim  , int(central_pixels_coord_tr_init[i , 0]) + half_dim + 1, int(central_pixels_coord_tr_init[i , 1]) - half_dim , int(central_pixels_coord_tr_init[i , 1]) + half_dim + 1]
        mask_reference_value = mask_padded[int(coordinates[0]) : int(coordinates[1]) , int(coordinates[2]) : int(coordinates[3])]
        last_reference_value = last_reference_padded[int(central_pixels_coord_tr_init[i, 0]) , int(central_pixels_coord_tr_init[i, 1])]
        actual_reference_value = actual_reference_padded[int(central_pixels_coord_tr_init[i, 0]) , int(central_pixels_coord_tr_init[i, 1])]
        test_pixels_indexs = np.transpose(np.array(np.where(mask_reference_value == 2)))
        if np.size(test_pixels_indexs,0) == 0:
            if (last_reference_value != 1 ) and (actual_reference_value <= 1):
                last_reference_patch = last_reference_padded[int(central_pixels_coord_tr_init[i, 0]) - half_dim : int(central_pixels_coord_tr_init[i, 0]) + half_dim + 1, int(central_pixels_coord_tr_init[i, 1]) - half_dim : int(central_pixels_coord_tr_init[i, 1]) + half_dim + 1]
                number_of_last_reference_pixels_indexs = np.where(last_reference_patch == 1)
                number_of_last_reference_pixels = np.size(number_of_last_reference_pixels_indexs , 1)
                porcent_of_last_reference = (number_of_last_reference_pixels/(patch_dimension*patch_dimension))*100
                if porcent_of_last_reference <= porcent_of_last_reference_in_actual_reference:
                    central_pixels_coord_tr[counter_tr, 0] = central_pixels_coord_tr_init[i , 0]
                    central_pixels_coord_tr[counter_tr, 1] = central_pixels_coord_tr_init[i , 1]
                    y_train_init[counter_tr, 0] = actual_reference_value
                    counter_tr += 1

    for i in range(np.size(central_pixels_coord_vl_init , 0)):
        coordinates = [int(central_pixels_coord_vl_init[i , 0]) - half_dim  , int(central_pixels_coord_vl_init[i , 0]) + half_dim + 1, int(central_pixels_coord_vl_init[i , 1]) - half_dim , int(central_pixels_coord_vl_init[i , 1]) + half_dim + 1]
        mask_reference_value = mask_padded[int(coordinates[0]) : int(coordinates[1]) , int(coordinates[2]) : int(coordinates[3])]
        last_reference_value = last_reference_padded[int(central_pixels_coord_vl_init[i, 0]) , int(central_pixels_coord_vl_init[i, 1])]
        actual_reference_value = actual_reference_padded[int(central_pixels_coord_vl_init[i, 0]) , int(central_pixels_coord_vl_init[i, 1])]
        test_pixels_indexs = np.transpose(np.array(np.where(mask_reference_value == 2)))
        if np.size(test_pixels_indexs,0) == 0:
            if (last_reference_value != 1 ) and (actual_reference_value <= 1):
                last_reference_patch = last_reference_padded[int(central_pixels_coord_vl_init[i, 0]) - half_dim : int(central_pixels_coord_vl_init[i, 0]) + half_dim + 1, int(central_pixels_coord_vl_init[i, 1]) - half_dim : int(central_pixels_coord_vl_init[i, 1]) + half_dim + 1]
                number_of_last_reference_pixels_indexs = np.where(last_reference_patch == 1)
                number_of_last_reference_pixels = np.size(number_of_last_reference_pixels_indexs , 1)
                porcent_of_last_reference = (number_of_last_reference_pixels/(patch_dimension*patch_dimension))*100
                if porcent_of_last_reference <= porcent_of_last_reference_in_actual_reference:
                    central_pixels_coord_vl[counter_vl, 0] = central_pixels_coord_vl_init[i , 0]
                    central_pixels_coord_vl[counter_vl, 1] = central_pixels_coord_vl_init[i , 1]
                    y_valid_init[counter_vl, 0] = actual_reference_value
                    counter_vl += 1

    return central_pixels_coord_tr, y_train_init, central_pixels_coord_vl, y_valid_init

def Central_Pixel_Definition_For_Test(mask, last_reference, actual_reference, patch_dimension, stride, mode):

    if mode == 'test':
        mask_rows = np.size(mask, 0)
        mask_cols = np.size(mask, 1)

        half_dim = patch_dimension//2
        upper_padding = np.zeros((half_dim, mask_cols))
        left_padding = np.zeros((mask_rows + half_dim, half_dim))
        bottom_padding = np.zeros((half_dim, half_dim + mask_cols))
        right_padding = np.zeros((2 * half_dim + mask_rows, half_dim))

        #Add padding to the mask
        mask_padded = np.concatenate((upper_padding, mask), axis=0)
        mask_padded = np.concatenate((left_padding, mask_padded), axis=1)
        mask_padded = np.concatenate((mask_padded, bottom_padding), axis=0)
        mask_padded = np.concatenate((mask_padded, right_padding), axis=1)

        #Add padding to the last reference
        last_reference_padded = np.concatenate((upper_padding, last_reference), axis=0)
        last_reference_padded = np.concatenate((left_padding, last_reference_padded), axis=1)
        last_reference_padded = np.concatenate((last_reference_padded, bottom_padding), axis=0)
        last_reference_padded = np.concatenate((last_reference_padded, right_padding), axis=1)

        #Add padding to the last reference
        actual_reference_padded = np.concatenate((upper_padding, actual_reference), axis=0)
        actual_reference_padded = np.concatenate((left_padding, actual_reference_padded), axis=1)
        actual_reference_padded = np.concatenate((actual_reference_padded, bottom_padding), axis=0)
        actual_reference_padded = np.concatenate((actual_reference_padded, right_padding), axis=1)

        mask = mask_padded
        last_reference = last_reference_padded
        actual_reference = actual_reference_padded

    #Initializing the central pixels coordinates containers
    central_pixels_coord_ts_init = []

    if stride == 1:
        central_pixels_coord_ts_init = np.where(mask == 2)
        central_pixels_coord_ts_init = np.transpose(np.array(central_pixels_coord_ts_init))
    else:
        print('[!] For test stride needs to be 1')

    #Refine the central pixels coordinates
    counter_ts = 0
    for i in range(np.size(central_pixels_coord_ts_init , 0)):
        last_reference_value = last_reference[int(central_pixels_coord_ts_init[i, 0]) , int(central_pixels_coord_ts_init[i, 1])]
        actual_reference_value = actual_reference[int(central_pixels_coord_ts_init[i, 0]) , int(central_pixels_coord_ts_init[i, 1])]
        if (last_reference_value != 1) and (actual_reference_value <= 1):
            counter_ts += 1

    central_pixels_coord_ts = np.zeros((counter_ts, 2))
    y_test_init = np.zeros((counter_ts,1))
    counter_ts = 0
    for i in range(np.size(central_pixels_coord_ts_init , 0)):
        last_reference_value = last_reference[int(central_pixels_coord_ts_init[i, 0]) , int(central_pixels_coord_ts_init[i, 1])]
        actual_reference_value = actual_reference[int(central_pixels_coord_ts_init[i, 0]) , int(central_pixels_coord_ts_init[i, 1])]
        if (last_reference_value != 1 ) and (actual_reference_value <= 1):
            central_pixels_coord_ts[counter_ts, 0] = central_pixels_coord_ts_init[i , 0]
            central_pixels_coord_ts[counter_ts, 1] = central_pixels_coord_ts_init[i , 1]
            y_test_init[counter_ts, 0] = actual_reference_value
            counter_ts += 1
                
    return central_pixels_coord_ts, y_test_init

def Classification_Maps(Predicted_labels, True_labels, central_pixels_coordinates, hit_map):

    Classification_Map = np.zeros((hit_map.shape[0], hit_map.shape[1], 3))
    TP_counter = 0
    FP_counter = 0
    for i in range(central_pixels_coordinates.shape[0]):

        T_label = True_labels[i]
        P_label = Predicted_labels[i]

        if T_label == 1:
            if P_label == T_label:
                TP_counter += 1
                #True positve
                Classification_Map[int(central_pixels_coordinates[i , 0]),int(central_pixels_coordinates[i , 1]),0] = 0
                Classification_Map[int(central_pixels_coordinates[i , 0]),int(central_pixels_coordinates[i , 1]),1] = 255
                Classification_Map[int(central_pixels_coordinates[i , 0]),int(central_pixels_coordinates[i , 1]),2] = 0
            else:
                #False Negative
                Classification_Map[int(central_pixels_coordinates[i , 0]),int(central_pixels_coordinates[i , 1]),0] = 255
                Classification_Map[int(central_pixels_coordinates[i , 0]),int(central_pixels_coordinates[i , 1]),1] = 255
                Classification_Map[int(central_pixels_coordinates[i , 0]),int(central_pixels_coordinates[i , 1]),2] = 0
        if T_label == 0:
            if P_label == T_label:
                #True Negative
                Classification_Map[int(central_pixels_coordinates[i , 0]),int(central_pixels_coordinates[i , 1]),0] = 255
                Classification_Map[int(central_pixels_coordinates[i , 0]),int(central_pixels_coordinates[i , 1]),1] = 255
                Classification_Map[int(central_pixels_coordinates[i , 0]),int(central_pixels_coordinates[i , 1]),2] = 255
            else:
                #False Positive
                FP_counter += 1
                Classification_Map[int(central_pixels_coordinates[i , 0]),int(central_pixels_coordinates[i , 1]),0] = 255
                Classification_Map[int(central_pixels_coordinates[i , 0]),int(central_pixels_coordinates[i , 1]),1] = 0
                Classification_Map[int(central_pixels_coordinates[i , 0]),int(central_pixels_coordinates[i , 1]),2] = 0

    return Classification_Map, TP_counter, FP_counter

def plot_embedding(X, y, d, title=None):
    """Plot an embedding X with the class label y colored by the domain d."""
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    # Plot colors numbers
    plt.figure(figsize=(10,10))
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        # plot colored number
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.bwr(d[i] / 1.),
                 fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)
