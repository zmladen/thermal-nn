from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from torchinfo import summary as ti_summary
import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
from torch.nn import Parameter as TorchParam
import random
from tqdm import trange
from typing import List, Tuple

class DataSet:
    input_cols = []
    target_cols = []
    dataset_path = None
    input_temperature_cols = []
    pid = "Not_Available"
    temperature_scale = 200  # in °C
    black_list = []  # some columns need to be dropped after featurizing

    def __init__(self, input_cols=None, target_cols=None):
        # make it possible to load more input cols than supposed by class definition
        input_cols = input_cols or self.input_cols
        target_cols = target_cols or self.target_cols
        self.data = pd.read_csv(self.dataset_path)
        col_arrangement = input_cols + [self.pid] + target_cols
        self.data = self.data.loc[:, [c for c in col_arrangement if c in self.data]]
        # note, some features in input/target cols will only exist after featurizing!
        self.input_cols = [c for c in input_cols if c in self.data]
        self.target_cols = [c for c in target_cols if c in self.data]

    @property
    def temperature_cols(self):
        return self.input_temperature_cols + self.target_cols

    @property
    def non_temperature_cols(self):
        return [c for c in self.data if c not in self.temperature_cols + [self.pid, 'train_'+self.pid]]

    def get_pid_sizes(self, pid_lbl=None):
        """Returns pid size as pandas Series"""
        pid_lbl = pid_lbl or self.pid
        return self.data.groupby(pid_lbl).agg('size').sort_values(ascending=False)

    def normalize(self):
        """Simple division by a scale, no offsets"""
        # Be wary that changing target_cols in featurize()
        #  and calling it after this normalize function will bring unexpected behavior
        #  e.g., adding a temperature to target cols in featurize and calling it after normalize will have
        #   that new target temperature normalized on its max value instead of temp_denom
        nt_cols = [c for c in self.non_temperature_cols if c in self.data]
        t_cols = [c for c in self.temperature_cols if c in self.data]
        # some columns might only exist after featurize()
        self.data.loc[:, t_cols] /= self.temperature_scale
        self.data.loc[:, nt_cols] /= self.data.loc[:, nt_cols].abs().max(axis=0)

    def get_profiles_for_cv(self, cv_lbl, kfold_split=4):
        """Given a cross-validation label and a table of profile sizes, return a tuple which associates
        training, validation and test sets with profile IDs.

        Args:
            cv_lbl (str): Cross-validation label. Allowed labels can be seen in wkutils.config.
            kfold_split (int, optional): The number of profiles per fold. Only active if cv_lbl=='kfold'. Defaults to 4.

        Returns:
            Tuple: training, validation and test set lists of lists of profile IDs fanned out by fold.
        """
        raise NotImplementedError()

class KaggleDataSet(DataSet):
    # This dataset is not using torque!!!
    # u_q,coolant,stator_winding,u_d,stator_tooth,motor_speed,i_d,i_q,pm,stator_yoke,ambient,torque,profile_id
    
    # input_cols = ['ambient', 'coolant', 'u_d', 'u_q', 'motor_speed', 'i_d', 'i_q']
    input_cols = ['u_q', 'coolant', 'u_d', 'motor_speed', 'i_d', 'i_q', 'ambient', 'torque']
    target_cols = ['pm', 'stator_yoke', 'stator_tooth', 'stator_winding']
    input_temperature_cols = ["ambient", "coolant"]
    # The following path might need to be replaced by the path on your system
    dataset_path = Path().cwd() / "data" / "input" / "measures_v2.csv"
    pid = "profile_id"
    name = "kaggle"
    sample_time = 0.5  # in seconds

    def get_profiles_for_cv(self, cv_lbl, kfold_split=4):
        """ Profiles for cross-validation"""

        pid_sizes = self.get_pid_sizes()
        if cv_lbl == '1fold':
            test_profiles = [[60, 62, 74]]
            validation_profiles = [[4]] # [[4]]
            train_profiles = [[p for p in pid_sizes.index.tolist() if p not in test_profiles + validation_profiles]]
        else:
            NotImplementedError(f"cv '{cv_lbl}' is not implemented.")

        # Use iloc when you want to select data based on its position in the DataFrame: df.iloc[1:5, 0:2]
        # Use loc when you want to select data based on its index or column names: df.loc['row1':'row3', 'col1':'col3']

        train_sample_size = pid_sizes.loc[test_profiles[0]].sum()
        print(f'Fold {0} test size: {train_sample_size} samples ' f'({train_sample_size / pid_sizes.sum():.1%} of total)')

        return train_profiles, validation_profiles, test_profiles

    def featurize(self):
        # extra feats (FE)
        # it is highly advisable to call featurize and then normalize, not the other way around!
        # Because featurize might mess with input and target cols
        if {'i_d', 'i_q', 'u_d', 'u_q'}.issubset(set(self.data.columns.tolist())):
            extra_feats = {'i_s': lambda x: np.sqrt((x['i_d']**2 + x['i_q']**2)),
                           'u_s': lambda x: np.sqrt((x['u_d']**2 + x['u_q']**2))}
        self.data = self.data.assign(**extra_feats).drop(columns=self.black_list)
        self.input_cols = [c for c in self.data if c not in self.target_cols + [self.pid]]
        # rearrange
        self.data = self.data.loc[:, self.input_cols + [self.pid] + self.target_cols]

class ChunkedKaggleDataSet(KaggleDataSet):
    name = "chunked_kaggle"

    def __init__(self, input_cols=None, target_cols=None, chunk_size=None):
        """Produce chunks/subsequences of each profile that act as new profiles with the same length.
        Effectively we have now much more profiles to train. Each profile has the same lenght. But if
        not enough data is available at the end the last chunk can be shorter.
        The total number of samples will be the same! Only number of profiles increases.
        """
        super().__init__(input_cols=input_cols, target_cols=target_cols)
        
        p_len = 1  # in hours
        chunk_size = chunk_size or int(p_len * 3600 / self.sample_time)
        tra_l, val_l, tst_l = self.get_profiles_for_cv(cv_lbl='1fold')

        # # Chunk only training datasets!!! 
        # # The test and validation is not cecessary. Alsp the profile_id should not be changed as 
        # # with this id the test and val tensors are defined.

        # # *******************************************
        # # Difficult to read but the same
        # # *******************************************
        # # tmp_profiles = [[df] if pid in val_l[0] + tst_l[0]
        # #                 else [df.iloc[n:min(n + chunk_size, len(df)), :].assign(**{self.pid: pid + i * 1000})
        # #                       for i, n in enumerate(range(0, len(df), chunk_size), start=1)]
        # #                 for pid, df in self.data.groupby(self.pid)]

        # tmp_profiles = []

        # # Loop through each group of data, grouped by Profile ID (pid)
        # for pid, df in self.data.groupby(self.pid):
            
        #     # Check if the current profile ID is in the validation or test sets
        #     if pid in val_l[0] + tst_l[0]:
        #         # If it is, we keep the data as it is (without chunking)
        #         tmp_profiles.append([df])
        #     else:
        #         # If it's not in the validation or test sets, we proceed to chunk the data

        #         # Initialize a list to store the chunks for this profile
        #         profile_chunks = []

        #         # Calculate the chunking points and loop through them
        #         for i, n in enumerate(range(0, len(df), chunk_size), start=1):
        #             # Define the chunk as a portion of the dataframe from 'n' to 'n + chunk_size'
        #             # Use min() to ensure we don't exceed the dataframe length
        #             chunk = df.iloc[n:min(n + chunk_size, len(df)), :]

        #             # Assign a new profile ID to each chunk
        #             new_pid = pid + i * 1000
        #             chunk = chunk.assign(**{self.pid: new_pid})

        #             # Add the chunk to the list of profile chunks
        #             profile_chunks.append(chunk)

        #         # Add all chunks of this profile to the main list
        #         tmp_profiles.append(profile_chunks)    

        # self.data = pd.concat([a for b in tmp_profiles for a in b], ignore_index=True)  # flatten

def generate_tensor(profiles_list, _ds, device, pid_lbl=None):
    """From the tabular data where all measurement profiles are concatenated on top of each other, create a 3D tensor"""
    pid_lbl = pid_lbl or _ds.pid

    if len(profiles_list) == 0:
        return None, None
    
    
    # there are possibly multiple pid columns due to chunked training set. Returns a list of labels.
    pid_lbls = [c for c in _ds.data if c.endswith(_ds.pid)]

    # tensor shape: (#time steps, #profiles, #features)
    tensor_shape = (
        _ds.get_pid_sizes(pid_lbl)[profiles_list].max(),  # Maximum size of profiles
        len(profiles_list),  # Number of profiles
        _ds.data.shape[1] - len(pid_lbls)  # Number of features, excluding profile ID labels
    )
    # Create an empty tensor filled with NaNs based on the determined shape
    tensor = np.full(tensor_shape, np.nan)

    # Populate the tensor with data
    # Iterate over each profile in 'profiles_list' after filtering and grouping the data by 'pid_lbl'
    for i, (pid, df) in enumerate(_ds.data.loc[_ds.data[pid_lbl].isin(profiles_list)].groupby(pid_lbl)):
        # Drop the columns that are profile ID labels from the DataFrame and convert the remaining data to a NumPy array
        profile_data = df.drop(columns=pid_lbls).to_numpy()
        tensor[:len(df), i, :] = profile_data


    sample_weights = 1 - np.isnan(tensor[:, :, 0])

    tensor = np.nan_to_num(tensor).astype(np.float32)
    tensor = torch.from_numpy(tensor).to(device)
    sample_weights = torch.from_numpy(sample_weights).to(device)
    return tensor, sample_weights

DEBUG = False
N_BATCHES = 42
CHUNK_SIZE = 1020

# Get the data and add new profiles based on the chunk size.
ds = ChunkedKaggleDataSet(chunk_size=CHUNK_SIZE)
ds.featurize()
ds.normalize()

input_cols = ds.input_cols
target_cols = ds.target_cols
temperature_cols = ds.temperature_cols
pid_sizes = ds.get_pid_sizes().to_dict()  # test and val set sizes
device = torch.device('cpu')


class DiffEqLayer(nn.Module):
    """This class is a container for the computation logic in each step. 
    This layer could be used for any 'cell', also RNNs, LSTMs or GRUs."""

    def __init__(self, cell, *cell_args):
        super().__init__()
        # self.cell = cell(*cell_args)
        self.cell = TNNCell()

    def forward(self, input: Tensor, state: Tensor) -> Tuple[Tensor,Tensor]:
        # input has shape [batch_size, 66, 10]
        inputs = input.unbind(0)

        # "unbind" results in batch_size number of tensors, each with the shape [66, 10].
        # This operation is useful when you want to process or handle each element in the batch separately. 
        # For example, if you are processing sequences or time series data where each sequence in the batch 
        # needs to be handled individually.

        outputs = torch.jit.annotate(List[Tensor], [])

        # outputs = torch.jit.annotate(List[Tensor], []) is essentially initializing an empty list named
        # outputs that is expected to contain PyTorch tensors, with this expectation explicitly communicated 
        # to TorchScript for optimization and serialization purposes. This is particularly useful when you 
        # need to dynamically build a list of tensors during the execution of your model, 
        # and you want this behavior to be correctly understood and handled by TorchScript during the scripting process.
        # Might be important for deploying PyTorch models in non-Python environments!

        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], state)
            outputs += [out]
        
        # stack: For example, if each tensor in outputs is of shape (A, B), and you are stacking N such tensors
        # without specifying the dimension, the resulting tensor will have the shape (N, A, B).
        # So output is a [batch_size, profile_number, features] -> [batch_size, 66, 10]
            
        return torch.stack(outputs), state
    
class TNNCell(nn.Module):
    """The main TNN logic. Here, the sub-NNs are initialized as well as the constant learnable
    thermal capacitances. The forward function houses the LPTN ODE discretized with the explicit Euler method"""

    def __init__(self):
        super().__init__()
        self.sample_time = 0.5  # in s
        self.output_size = len(target_cols) # 4 temperatures are to be predicted!

        # self.caps shape: [4] it is a 1D tensor!

        # Torch parameters are simple coefficients that are trained. Allow custom layers to be defined? 
        self.caps = TorchParam(torch.Tensor(self.output_size).to(device))
        nn.init.normal_(self.caps, mean=-9.2, std=0.5)  # hand-picked init mean, might be application-dependent

        # In a network of n nodes, the number of unique pairwise connections is given by the formula n×(n−1)/2.
        # This is because each node can be connected to every other node, but since connections are bidirectional
        # (i.e., the connection from node A to node B is the same as from B to A), you only count half of the total possible connections.
        # n combinatorics, the number of combinations of items taken 2 at a time from n items is denoted as (n 2). Which reduced to n×(n−1)/2.

        n_temps = len(temperature_cols) # Number of nodes -> 6 one for each temperature!
        n_conds = int(0.5 * n_temps * (n_temps - 1)) # Number of unique connections for conduntances -> 15 different connections!



        # Why Include output_size in Inputs:
        # In many complex systems, especially those involving physical processes like thermal networks,
        # the relationships between variables are not just one-directional (from input to output).
        # Instead, the state of the output can influence the behavior of the system, which in turn affects future states.
        # For instance, in a thermal system, the current temperature at various nodes (which can be part of self.output_size)
        # could affect how heat conducts between these nodes. 
        # Hence, it makes sense to include information about these temperatures as part of the input to the conductance_net.

        # conductance net sub-NN
        # self.conductance_net get an input of 10 (features) + 4 (estimates) and outputs 4 conductances!
        self.conductance_net = nn.Sequential(nn.Linear(len(input_cols) + self.output_size, n_conds), nn.Sigmoid())

        # Populate adjacency matrix. It is used for indexing the conductance sub-NN output

        # Matrix:
        # | a11  a12  a13 a14 a15 a16 |
        # | a21  a22  a23 a24 a25 a26 |
        # | a31  a32  a33 a34 a35 a36 |
        # | a41  a42  a43 a44 a45 a46 |
        # | a51  a52  a53 a54 a55 a56 |
        # | a61  a52  a63 a64 a65 a66 |

        self.adj_mat = np.zeros((n_temps, n_temps), dtype=int)
        adj_idx_arr = np.ones_like(self.adj_mat)

        # Indices of upper triangle with main diagonal:
        # | (0,0)  (0,1)  (0,2) (0,3) (0,4) (0,5) |
        # |        (1,1)  (1,2) (1,3) (1,4) (1,5) |
        # |               (2,2) (2,3) (2,4) (2,5) |
        # |                     (3,3) (3,4) (3,5) |
        # |                           (4,4) (4,5) |
        # |                                 (5,5) |
    
        # Indices of upper triangle with 1 offset from main diagonal:
        # |        (0,1)  (0,2) (0,3) (0,4) (0,5) |
        # |               (1,2) (1,3) (1,4) (1,5) |
        # |                     (2,3) (2,4) (2,5) |
        # |                           (3,4) (3,5) |
        # |                                 (4,5) |


        # triu_indices returns 2 arrays:
        # 1. (array([0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 4])) represents the row indices. 
        # 2. (array([1, 2, 3, 4, 5, 2, 3, 4, 5, 3, 4, 5, 4, 5, 5])) represents the column indices.

        triu_idx = np.triu_indices(n_temps, 1)

        # Now extract elements of the upper triangular excluding main diagonal to array:
        # [a12, a13, a14, a15, a16, a23, a24, a25, a26, a34, a35, a36, a45, a46, a56]
        adj_idx_arr = adj_idx_arr[triu_idx].ravel()
        
        # This step is important to correctly index the conductances (heat transfer coefficients) between each pair of nodes.
        # This step is assigning these calculated indices to the upper triangular part of the adjacency matrix. 
        # This essentially maps each pair of connected nodes to a specific index in the list of conductances.#
        self.adj_mat[triu_idx] = np.cumsum(adj_idx_arr) - 1
        # Makes the matrix symmetric. As connections of conductances are bidirectional
        self.adj_mat += self.adj_mat.T

        # Result is:
        # | 0  0  1  2  3  4  |
        # | 0  0  5  6  7  8  |
        # | 1  5  0  9  10 11 |
        # | 2  6  9  0  12 13 |
        # | 3  7  10 12 0  14 |
        # | 4  8  11 13 14 0  |

 
        # Crop the matrix to have the needed form of [4, 6]

        # tensor([[ 0,  0,  1,  2,  3,  4],
        #         [ 0,  0,  5,  6,  7,  8],
        #         [ 1,  5,  0,  9, 10, 11],
        #         [ 2,  6,  9,  0, 12, 13]])



        self.adj_mat = torch.from_numpy(self.adj_mat[:self.output_size, :]).type(torch.int64) # crop
        self.n_temps = n_temps
        
        # power loss sub-NN
        # the same as conductance_nn.
        self.ploss = nn.Sequential(nn.Linear(len(input_cols) + self.output_size, 16),
                                    nn.Tanh(),
                                    nn.Linear(16, self.output_size),
                                   )
        
        # How these indices are used? Need to understand!!! They are indexes of measured coolant and ambient temperatures!
        self.temp_idcs = [i for i, x in enumerate(input_cols) if x in temperature_cols]
        self.nontemp_idcs = [i for i, x in enumerate(input_cols) if x not in temperature_cols + ['profile_id']]

        # temp_idcs
        # [1, 6]
        # nontemp_idcs
        # [0, 2, 3, 4, 5, 7, 8, 9]

    def forward(self, inp: Tensor, hidden: Tensor) -> Tuple[Tensor, Tensor]:  
        # inp are features that are measurable!
        # ANN outputs dependent on 𝜻[𝑘] = [𝝑̃a[k] 𝝑̂[k] 𝝃[k]], which in turn consists of the ancillary temperatures 𝝑̃a[𝑘],
        # the temperature estimates 𝝑̂[k] and additional observables 𝝃[𝑘] for each sample.

        # input_cols:           ['u_q', 'coolant', 'u_d', 'motor_speed', 'i_d', 'i_q', 'ambient', 'torque', 'i_s', 'u_s']
        # temperature_cols      ['pm', 'stator_yoke', 'stator_tooth', 'stator_winding', 'ambient', 'coolant']
        # target_cols           ['pm', 'stator_yoke', 'stator_tooth', 'stator_winding']
        # sub_nn_inp 𝜻[𝑘]:      ['u_q', 'coolant', 'u_d', 'motor_speed', 'i_d', 'i_q', 'ambient', 'torque', 'i_s', 'u_s', 'pm', 'stator_yoke', 'stator_tooth', 'stator_winding']
        # prev_out / hidden     ['pm', 'stator_yoke', 'stator_tooth', 'stator_winding']

        prev_out = hidden

        # Add measured temperatures to the estimated temperatures, shape is [66 ,6]
        temps = torch.cat([prev_out, inp[:, self.temp_idcs]], dim=1)

        sub_nn_inp = torch.cat([inp, prev_out], dim=1) # this adds prev_out to inp as new columns to form sub_nn_inp 𝜻[𝑘]

        conducts = torch.abs(self.conductance_net(sub_nn_inp))

        # Conducts has shape of [66, 15]. Which calculates all 15 contuctances for each profile.
        # the adj_mat is used to transform these 15 values to a [4, 6] matrix. Values in the adj_mat are the indexes of the conducts array!
        # Matrix form is better since the multiplication can be parallelized using tensors operations! Otherwise for-loops muzst be used
        # which is computationally inefficient! 
        
        power_loss = torch.abs(self.ploss(sub_nn_inp))

        # inp shape:                [66, 10]
        # prev_out shape:           [66 4]
        # hidden shape:             [66 4]
        # adj_mat shape             [4, 6]
        # conducts shape:           [66, 15]
        # power loss shape:         [66, 4]
        # sub_nn_inp shape:         [66, 14]
        # temps shape:              [66, 6]

        # print("conducts shape")
        # print(conducts.shape)
        # print("power_loss shape")
        # print(power_loss.shape)
        # print("sub_nn_inp shape:")
        # print(sub_nn_inp.shape)
        # print("self.adj_mat shape")
        # print(self.adj_mat.shape)
        # print("inp shape:")
        # print(inp.shape)
        # print("temps.shape")
        # print(temps.shape)
        # print("prev_out shape:")
        # print(prev_out.shape)


        # We need for each profile:
        # | (a1-a1)*c11 + (a1-a2)*c12 + (a1-a3)*c13 + (a1-a4)*c14 + (a1-a5)*c14 + (a1-a6)*c14 |
        # | (a2-a1)*c21 + (a2-a2)*c22 + (a2-a3)*c23 + (a2-a4)*c24 + (a2-a5)*c14 + (a2-a6)*c14 |
        # | (a3-a1)*c31 + (a3-a2)*c32 + (a3-a3)*c33 + (a3-a4)*c34 + (a3-a5)*c14 + (a3-a6)*c14 |
        # | (a4-a1)*c41 + (a4-a2)*c42 + (a4-a3)*c43 + (a4-a4)*c44 + (a4-a5)*c14 + (a4-a6)*c14 |

    	# This can be done using brodcasting with these vectors, also per profile:

        #             v1              -     v2    =                       r

        # | a1, a2, a3, a4, a5, a6 |      | a1 |      | a1-a1, a2-a1, a3-a1, a4-a1, a5-a1, a6-a1 |
        #                                 | a2 |      | a1-a2, a2-a2, a3-a2, a4-a2, a5-a2, a6-a2 |
        #                             -   | a3 |  =   | a1-a3, a2-a3, a3-a3, a4-a3, a5-a3, a6-a3 |
        #                                 | a4 |      | a1-a4, a2-a4, a3-a4, a4-a4, a5-a4, a6-a4 |
            
        # c:
        # | c11 c12 c13 c14 c15 c16 |
        # | c21 c22 c23 c24 c25 c26 |
        # | c31 c32 c33 c34 c35 c36 |
        # | c41 c42 c43 c44 c45 c46 |

        # final result is elementwise multiplication r * c
        # Teen all needs to be summed.
         
        # temps:                       [66, 6]
        # prev_out:                    [66, 4]
        # as a result we need:         [66, 4, 6]
        # so for broadcasting we need: [66, 1, 6] and [66, 4, 1]
        # temp.unsqueeze(1)            [66, 1, 6]
        # prev_out.unsqueeze(-1):      [66, 4, 1]

        # This can be easier understood if first dimenstion is neglected. Thus we observe vectors [1, 6] and [4, 1]

        # After multiplication the r * c has shape of [66, 4 ,6]        
        temp_diffs = torch.sum((temps.unsqueeze(1) - prev_out.unsqueeze(-1)) * conducts[:, self.adj_mat], dim=-1)
        # After summation temp_diffs has shape of [66, 4]

        # ********************************************************************************************
        # The same can be achieved with foor loops. This is however not computationaly not efficient!
        # ********************************************************************************************

        # temp_diffs = torch.zeros_like(prev_out)  # Shape [66, 4]
        # # Iterate over each profile/time step
        # for k in range(temps.shape[0]):  # Looping over 66 profiles
        #     # Iterate over each internal node
        #     for i in range(prev_out.shape[1]):  # Looping over 4 internal nodes
        #         sum_diffs = 0
        #         # Iterate over all nodes (both internal and external)
        #         for j in range(temps.shape[1]):  # Looping over 6 nodes (internal + external)
        #             # Calculate temperature difference
        #             temp_diff = temps[k, j] - prev_out[k, i]
        #             # Get the corresponding conductance value
        #             conductance = conducts[k, self.adj_mat[i, j]]
        #             # Add the product of the difference and the conductance to the sum
        #             sum_diffs += temp_diff * conductance
        #         # Assign the sum to the corresponding element in temp_diffs
        #         temp_diffs[k, i] = sum_diffs

        # ********************************************************************************************
        # end
        # ********************************************************************************************

        # [66, 4] = [66,4] + 0.5 * [4] * ([66 4] + [66, 4])

        # To multiply [4] *  [66, 4] broadcasting is applied. Similar to substraction!
        # vector [4] is coppies 66 times to make new vector [66, 4]. Each row is the same! Then [66, 4] and [66, 4] is multiplied elementwise.
        out = prev_out + self.sample_time * torch.exp(self.caps) * (temp_diffs + power_loss)
        
        # torch.exp(self.caps): [4]
        # out:                  [66, 4]
        # temp_diffs:           [66, 4]
        # power_loss:           [66, 4]
        # sample time:          [1]
        # prev_out:             [66, 4]

        # print("temp_diffs shape")
        # print(temp_diffs.shape)
        # print("torch.exp(self.caps) shape")
        # print(torch.exp(self.caps).shape)
        # print("shape out")


        return prev_out, torch.clip(out, -1, 5) # will ensure that every element in the tensor out falls within the range -1 to 5


# Training parameters
train_l, val_l, test_l = ds.get_profiles_for_cv("1fold")

# test_l [[60, 62, 74]]
# val_l [[4]]
# train_l [[..., 36013, 26070, 23036, 7051, 3046, 3047, 7064, 15057, 44020, 1606]]

# A dictionary to store various logs like training and validation loss trends and model state dictionaries.
# This is useful for monitoring the model's performance and debugging.

logs = {'loss_trends_train': [[] for _ in train_l],
        'loss_trends_val': [[] for _ in val_l],
        'models_state_dict': [],
        }

# Random number generation is set to ensure reproducibility.
SEED = 0
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

n_epochs = 100

# train_tensor, train_sample_weights = generate_tensor(train_l[0], ds, device, pid_lbl=ds.pid)
# print(train_tensor)

# Training loop for each fold
for fold_i, (fold_train_profiles, fold_val_profiles, fold_test_profiles) in enumerate(zip(train_l, val_l, test_l)):
    # generate tensors
    train_tensor, train_sample_weights = generate_tensor(fold_train_profiles, ds, device, pid_lbl=ds.pid)
    val_tensor, val_sample_weights = generate_tensor(fold_val_profiles or [], ds, device)
    test_tensor, test_sample_weights = generate_tensor(fold_test_profiles, ds, device)
    
    # Truncated Backpropagation Through Time (TBPTT) size - how many timesteps to propagate the error back.
    tbptt_size =  256 #128

    # Calculating the number of batches based on the size of your training data and the TBPTT size.
    n_batches = np.ceil(train_tensor.shape[0] / tbptt_size).astype(int)

    # Initialize the model and optimizer
    model = torch.jit.script(DiffEqLayer(TNNCell).to(device))
    loss_func = nn.MSELoss(reduction="none")
    opt = optim.Adam(model.parameters(), lr=1e-3)

    pbar = trange(n_epochs, desc=f"Seed {SEED}, fold {fold_i}", position=fold_i, unit="epoch")
    if fold_i == 0:  # print only once
        mdl_info = ti_summary(model, device=device, verbose=0)
        pbar.write(str(mdl_info))
        logs['model_size'] = mdl_info.total_params

    # it is important to transfer model to CPU right here, after model_stats were printed
    # otherwise, one model in a process might get back to GPU, whysoever
    model.to(device)
    
    # generate shuffled indices in before hand
    idx_mat = []
    for i in range(n_epochs):
        idx = np.arange(train_tensor.shape[1])
        np.random.shuffle(idx)
        idx_mat.append(idx)
    idx_mat = np.vstack(idx_mat)


    # Initialize the variable to keep track of the lowest loss.
    lowest_loss = float('inf')

    # Initialize a list to store loss values
    epoch_losses = []

    # Enable interactive mode for matplotlib
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title('Loss Curve')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    line, = ax.plot(epoch_losses, label='Training Loss')
    ax.legend()


    start_time = pd.Timestamp.now().round(freq='S')
    for epoch in pbar:    
        # shuffle profiles
        indices = idx_mat[epoch]
        train_tensor_shuffled = train_tensor#[:, indices, :]
        train_sample_weights_shuffled = train_sample_weights#[:, indices]
        
        hidden = train_tensor_shuffled[0, :,  -len(target_cols):]

        for n in range(n_batches):
            # Zero the gradients before a new step of optimization.
            model.zero_grad()

            # Forward pass: compute the model output for the current batch.
            output, hidden = model(train_tensor_shuffled[i*tbptt_size:(i+1)*tbptt_size, :, :len(ds.input_cols)], hidden.detach())

            # Calculate loss: comparing the model output with the target values.
            train_loss = loss_func(output, train_tensor_shuffled[i*tbptt_size:(i+1)*tbptt_size, :, -len(ds.target_cols):])
            
            # Apply sample weighting to the loss.
            train_loss = (train_loss * train_sample_weights_shuffled[i*tbptt_size:(i+1)*tbptt_size, :, None] / train_sample_weights_shuffled[i*tbptt_size:(i+1)*tbptt_size, :].sum()).sum().mean()

            # Backward pass: compute gradient of the loss with respect to model parameters.
            train_loss.backward()
            # Perform a single optimization step (parameter update).
            opt.step()

        # Append the loss of the current epoch to the list
        epoch_losses.append(train_loss.item())

        # Checkpointing: Save the model if the current loss is the lowest.
        current_loss = train_loss.item()
        if current_loss < lowest_loss:
            lowest_loss = current_loss
            torch.save(model.state_dict(), 'model_with_lowest_loss_fold.pth')

        # Update the plot
        line.set_xdata(np.arange(len(epoch_losses)))
        line.set_ydata(epoch_losses)
        ax.relim()  # Recalculate limits
        ax.autoscale_view(True, True, True)  # Rescale the view
        plt.pause(0.01)  # Pause to update the plot


        with torch.no_grad():
            logs["loss_trends_train"][fold_i].append(train_loss.item())
            pbar_str = f'Loss {train_loss.item():.2e}'

        # validation set
        if val_tensor is not None:
            with torch.no_grad():
                pred, hidden = model(val_tensor[:, :, :len(input_cols)], val_tensor[0, :,  -len(target_cols):])
                # logging

                # Calculate loss: comparing the model output with the target values.
                val_loss = loss_func(pred, val_tensor[:, :,  -len(target_cols):])
                # Apply sample weighting to the loss.
                val_loss = (val_loss * val_sample_weights[:, :, None] / val_sample_weights[:, :].sum()).sum().mean()

                logs["loss_trends_val"][fold_i].append(val_loss.item())
                pbar_str += f'| val loss {val_loss.item():.2e}'

        pbar.set_postfix_str(pbar_str)


    # Evaluate on test set
    # ...

    # Save model state
    logs["models_state_dict"].append(model.state_dict())

# Additional steps for logging and analysis
# ...