import pandas as pd
import numpy as np
import os
import pickle
from tqdm import tqdm
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset
import multiprocessing


class Input:
    # Class variables (shared across instances)
    path = ''
    mode = ''
    dataset_name = ''
    prefix_len = ''
    batch = ''
    design_matrix = ''
    design_matrix_padded = ''
    y = ''
    unique_event = ''
    selected_columns = ''
    timestamp_loc = ''
    train_inds = ''
    test_inds = ''
    validation_inds = ''
    train_loader = ''
    test_loader = ''
    validation_loader = ''

    @classmethod
    def run(cls, path, prefix, batch_size, mode="event_prediction"):
        cls.prefix_len = prefix
        cls.batch = batch_size
        cls.dataset_name = path.split("/")[-1].split('.')[0]
        cls.mode = mode
        cls.path = os.path.join("outputs", cls.dataset_name, mode, f"prefix_{cls.prefix_len}")

        # Reading a file
        if path.split('.')[1] == 'csv':
            data_augment = cls.__read_csv_massive(path)
        elif path.split('.')[1] == 'pkl':
            data_augment = pickle.load(open(path, "rb"))
            print("The head of augmented with remaining and duration times:\n", data_augment.head(10))
        else:
            raise ValueError("Unsupported file format")

        # Creating a design matrix that shows one hot vector representation for activity IDs
        cls.design_matrix = cls.__design_matrix_creation(data_augment)

        # Creating prefix
        cls.prefix_creating(prefix, mode)

        # Determining the train, test, and validation sets
        cls.train_valid_test_index()

        # Creating minibatch
        cls.mini_batch_creation(batch_size)

    #################################################################################
    @classmethod
    def __read_csv(cls, path):
        dat = pd.read_csv(path)
        print("Types:", dat.dtypes)

        dat['ActivityID'] = dat['ActivityID'].astype('category')
        dat['CompleteTimestamp'] = dat['CompleteTimestamp'].astype('datetime64[ns]')
        print("Types after:", dat.dtypes)

        print("columns:", dat.columns)
        dat_group = dat.groupby('CaseID')
        print("Original data:", dat.head())
        print("Group by data:", dat_group.head())

        data_augment_list = []
        dat_group = dat.groupby('CaseID')

        total_iter = len(dat_group.ngroup())
        pbar = tqdm(total=total_iter)

        for _, gr in dat_group:
            gr = gr.sort_values(by=['CompleteTimestamp'])

            duration_time = gr.loc[:, 'CompleteTimestamp'].diff() / np.timedelta64(1, 'D')
            duration_time.iloc[0] = 0

            length = duration_time.shape[0]
            remaining_time = [np.sum(duration_time[i + 1:length]) for i in range(length)]

            gr['duration_time'] = duration_time
            gr['remaining_time'] = remaining_time

            data_augment_list.append(gr)
            pbar.update(1)

        pbar.close()
        data_augment = pd.concat(data_augment_list, ignore_index=True)

        name = path.split(".")[0].split("/")[-1]
        pickle.dump(data_augment, open(name + ".pkl", "wb"))

        print("Dataset with indicating remaining and duration times:\n", data_augment.head(10))
        return data_augment

    ################################################################################
    @classmethod
    def __read_csv_massive(cls, path):
        dat = pd.read_csv(path)
        print("Types:", dat.dtypes)

        dat['ActivityID'] = dat['ActivityID'].astype('category')
        dat['CompleteTimestamp'] = dat['CompleteTimestamp'].astype('datetime64[ns]')
        print("Types after:", dat.dtypes)

        print("columns:", dat.columns)
        print("Original data:", dat.head())

        num_processes = multiprocessing.cpu_count()
        chunk_size = int(dat.shape[0] / num_processes)
        chunks = [dat.iloc[dat.index[i:i + chunk_size]] for i in range(0, dat.shape[0], chunk_size)]

        pool = multiprocessing.Pool(processes=num_processes)
        results = pool.map(cls.func, chunks)
        pool.close()
        pool.join()

        results = pd.concat(results, ignore_index=True)

        name = os.path.splitext(os.path.basename(path))[0]
        pickle.dump(results, open(os.path.join("data", name + ".pkl"), "wb"))

        return results

    ######################################################################################
    @classmethod
    def func(cls, dat):
        data_augment_list = []
        dat_group = dat.groupby('CaseID')

        total_iter = len(dat_group.ngroup())
        pbar = tqdm(total=total_iter)

        for _, gr in dat_group:
            gr = gr.sort_values(by=['CompleteTimestamp'])

            duration_time = gr.loc[:, 'CompleteTimestamp'].diff() / np.timedelta64(1, 'D')
            duration_time.iloc[0] = 0

            length = duration_time.shape[0]
            remaining_time = [np.sum(duration_time[i + 1:length]) for i in range(length)]

            gr['duration_time'] = duration_time
            gr['remaining_time'] = remaining_time

            data_augment_list.append(gr)
            pbar.update(1)

        pbar.close()
        return pd.concat(data_augment_list, ignore_index=True)

    #######################################################################################
    @classmethod
    def __design_matrix_creation(cls, data_augment):
        unique_event = sorted(data_augment['ActivityID'].unique())
        cls.unique_event = [0] + unique_event
        print("unique events:", unique_event)

        l = []
        for _, row in tqdm(data_augment.iterrows(), total=len(data_augment)):
            temp = {k: 1 if k == row['ActivityID'] else 0 for k in ['0'] + list(unique_event)}
            temp['class'] = row['ActivityID']
            temp['duration_time'] = row['duration_time']
            temp['remaining_time'] = row['remaining_time']
            temp['CaseID'] = row['CaseID']
            l.append(temp)

        design_matrix = pd.DataFrame(l)
        print("The design matrix is:\n", design_matrix.head(10))
        return design_matrix

    ################################################################################
    @classmethod
    def prefix_creating(cls, prefix=2, mode='event_prediction'):
        if mode == "timestamp_prediction":
            clsN = cls.design_matrix.columns.get_loc('duration_time')
        elif mode == "event_prediction":
            clsN = cls.design_matrix.columns.get_loc('class')
        elif mode == 'event_timestamp_prediction':
            clsN = [
                cls.design_matrix.columns.get_loc('duration_time'),
                cls.design_matrix.columns.get_loc('class')
            ]
            cls.timestamp_loc = cls.design_matrix.columns.get_loc('duration_time')
            cls.selected_columns = cls.unique_event + [cls.timestamp_loc]

        group = cls.design_matrix.groupby('CaseID')
        temp, temp_shifted = [], []

        for _, gr in group:
            gr = gr.drop('CaseID', axis=1).reset_index(drop=True)

            new_row = [0] * gr.shape[1]
            gr.loc[gr.shape[0]] = new_row
            gr.iloc[gr.shape[0] - 1, gr.columns.get_loc('0')] = 1

            gr_shift = gr.shift(periods=-1, fill_value=0)
            gr_shift.loc[gr.shape[0] - 1, '0'] = 1

            if (gr.shape[0] - 1) > prefix:
                for i in range(gr.shape[0]):
                    temp.append(torch.tensor(gr.iloc[i:i + prefix].values, dtype=torch.float, requires_grad=False))
                    try:
                        temp_shifted.append(
                            torch.tensor([gr.iloc[i + prefix, clsN]], dtype=torch.float, requires_grad=False))
                    except IndexError:
                        temp_shifted.append(torch.tensor([np.float16(0)], dtype=torch.float, requires_grad=False))

        cls.design_matrix_padded = pad_sequence(temp, batch_first=True)
        cls.y = pad_sequence(temp_shifted, batch_first=True)
        cls.__pad_correction()

        print("The dimension of designed matrix:", cls.design_matrix_padded.size())
        print("The dim of ground truth:", cls.y.size())
        print("The prefix considered so far:", cls.design_matrix_padded.size()[1])

    ########################################################################################
    @classmethod
    def __pad_correction(cls):
        for i in range(cls.design_matrix_padded.size()[0]):
            u = (cls.design_matrix_padded[i, :, 0] == 1).nonzero()
            try:
                cls.design_matrix_padded[i, :, 0][u:] = 1
            except TypeError:
                pass

    ##################################################################################
    @classmethod
    def train_valid_test_index(cls):
        train_inds = np.arange(0, round(cls.design_matrix_padded.size()[0] * .8))
        test_inds = list(set(range(cls.design_matrix_padded.size()[0])).difference(set(train_inds)))
        validation_inds = test_inds[0:round(0.3 * len(test_inds))]
        test_inds = test_inds[round(0.3 * len(test_inds)):]

        cls.train_inds = train_inds
        cls.test_inds = test_inds
        cls.validation_inds = validation_inds

        print("Number of training instances:", len(train_inds))
        print("Number of testing instances:", len(test_inds))
        print("Number of validation instances:", len(validation_inds))

    #################################################################################
    @classmethod
    def testData_correction(cls):
        test_inds_new = []
        for i in cls.test_inds:
            u = (cls.design_matrix_padded[i, :, 0] == 1).nonzero()
            if len(u) <= 1:
                test_inds_new.append(i)

        print("The number of test prefixes before correction:", len(cls.test_inds))
        print("The number of test prefixes after correction:", len(test_inds_new))
        cls.test_inds = test_inds_new

    #################################################################################
    @classmethod
    def mini_batch_creation(cls, batch=4):
        train_data = TensorDataset(cls.design_matrix_padded[cls.train_inds], cls.y[cls.train_inds])
        cls.train_loader = DataLoader(dataset=train_data, batch_size=batch, shuffle=True)

        test_data = TensorDataset(cls.design_matrix_padded[cls.test_inds], cls.y[cls.test_inds])
        cls.test_loader = DataLoader(dataset=test_data, batch_size=batch, shuffle=True)

        validation_data = TensorDataset(cls.design_matrix_padded[cls.validation_inds], cls.y[cls.validation_inds])
        cls.validation_loader = DataLoader(dataset=validation_data, batch_size=batch, shuffle=True)

        print("The minibatch is created!")
