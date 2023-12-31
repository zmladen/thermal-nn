import os
import argparse
import time
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
try:
    plt.style.use('seaborn-v0_8-talk')
except OSError:
    plt.style.use("seaborn-talk")

from torchinfo import summary as ti_summary
import torch
import torch.nn as nn
import torch.optim as optim
import torch.jit as jit
from torch.nn import Parameter as TorchParam
from torch import Tensor
from typing import List, Tuple
from torch.utils.tensorboard import SummaryWriter


# Read the data. All recordings are sampled at 2 Hz.

path_to_csv=Path().cwd()/ 'data' / 'input' / 'measures_v2.csv'
data = pd.read_csv(path_to_csv)
target_cols = ['pm', 'stator_yoke', 'stator_tooth', 'stator_winding']

temperature_cols = target_cols + ['ambient', 'coolant']

test_profiles = [4, 67, 71, 78] # [60, 62, 74]
train_profiles = [p for p in data.profile_id.unique() if p not in test_profiles]

# Count the number of rows / datapoints for each unique 'profile_id' 
profile_sizes = data.groupby('profile_id').agg('size')

# Normalize
non_temperature_cols = [c for c in data if c not in temperature_cols + ['profile_id']]
data.loc[:, temperature_cols] /= 200 # deg C
data.loc[:, non_temperature_cols] /= data.loc[:, non_temperature_cols].abs().max(axis=0)

# Feature engineering
# A set is a data structure in Python that only contains unique elements.
# Using a set here effectively removes any duplicate column names that might exist in the DataFrame. 
# set is {"A", "B", "C", ...}

if {'i_d', 'i_q', 'u_d', 'u_q'}.issubset(set(data.columns.tolist())):
    extra_feats = {'i_s': lambda x: np.sqrt((x['i_d']**2 + x['i_q']**2)), 
                   'u_s': lambda x: np.sqrt((x['u_d']**2 + x['u_q']**2))}

    # # Calculate 'i_s' and 'u_s' directly without checking column existence
    # data['i_s'] = np.sqrt(data['i_d']**2 + data['i_q']**2)
    # data['u_s'] = np.sqrt(data['u_d']**2 + data['u_q']**2)

data = data.assign(**extra_feats)

# Rearrange features in order input_cols, profile_id and target_cols
# input_cols are all columns except target cplumns to be estimated
input_cols = [c for c in data.columns if c not in target_cols + ['profile_id']]
data = data.loc[:, input_cols + ['profile_id'] + target_cols]

#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# overwrite. We recommend CPU over GPU here, as that runs faster with pytorch on this data set
device = torch.device('cpu')

def generate_tensor(profiles_list):
    # Create an empty tensor filled with NaNs. Its shape is based on the maximum size of profiles, 
    # number of profiles, and the number of columns in the data minus one ('profile_id' is not used for NN).
    # In time or sequence models the input tensor is typically organized with dimensions representing
    # time steps, sequence identifiers, and features!!!

    tensor = np.full((profile_sizes[profiles_list].max(), len(profiles_list), data.shape[1] - 1), np.nan)

    # Filter the data to only include rows where the profile ID is in profiles_list.
    filtered_data = data[data.profile_id.isin(profiles_list)]

    # Group this filtered data by each unique profile ID so it can be iterated.
    grouped_data = filtered_data.groupby('profile_id')

    # Now, iterate over each group and populate the tensor.
    for index, (profile_id, profile_data) in enumerate(grouped_data):
        # Check if the profile ID is in the list of profiles. If not, show an error message.
        if profile_id not in profiles_list:
            raise ValueError(f"Profile ID {profile_id} is not in the list: {profiles_list}")

        # Populate tensor with profile data
        tensor[:len(profile_data), index, :] = profile_data.drop(columns='profile_id').to_numpy()

    # Handling Missing Data in Models: When training machine learning models, especially neural networks,
    # it's common to encounter datasets with missing values. Directly feeding data with missing values into a model
    # can lead to poor performance or inaccurate results.

    # Weighting Samples in Training: sample_weights can be used to weight the importance of each sample during model training.
    # For example, in your case, samples with missing data (marked by 0 in sample_weights) can be given less importance or ignored
    # during the training process. This approach helps in focusing the model's learning on samples with complete data.
    # The loss will be multiplied elementwise with this matrix, making the loss zero for none existing data!
        
    sample_weights = 1 - np.isnan(tensor[:, :, 0])

    # Replace NaNs with zero and convert the tensor to float32 data type.
    tensor = np.nan_to_num(tensor).astype(np.float32)

    # Convert the numpy tensor to a PyTorch tensor and send it to the specified device (GPU/CPU).
    tensor = torch.from_numpy(tensor).to(device)
    sample_weights = torch.from_numpy(sample_weights).to(device)

    # tensor: torch.Size([43971, 66, 14])
    # sample_weights: torch.Size([43971, 66])

    # sample_weights[:,0] all weights of the first profile.
    # sample_weights[:,0] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    # Wherever data is present is has 1 otherwise it is 0. The loss is devided but the sums of 1 to weight the 
    # importance of the particular loss!
    # Without this normalization step, batches with more valid samples would naturally have a higher
    # total loss simply due to the larger number of samples, and not necessarily due to worse model performance.
    # Dividing by the sum of weights(train_sample_weights[i*tbptt_size:(i+1)*tbptt_size, :].sum()) scales the loss to the average per contributing sample, 
    # making it a more meaningful metric.

    return tensor, sample_weights

def plot_features(tensor):
    # Determine the number of profiles and features from the tensor's shape
    num_profiles, num_features = tensor.shape[1], tensor.shape[2]

    for feature_index in range(num_features):
        # Determine the number of rows and columns for the subplots
        num_rows = int(np.ceil(np.sqrt(num_profiles)))
        num_cols = int(np.ceil(num_profiles / num_rows))

        # Create a figure with subplots
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 10))
        fig.suptitle(f'Feature {feature_index + 1} Across Different Profiles')

        # Flatten the axes array for easy iteration
        axes = axes.flatten()

        # Loop through each profile
        for profile_index in range(num_profiles):
            # Select the data for the current profile and feature
            profile_feature_data = tensor[:, profile_index, feature_index]

            # Plot the data on the respective subplot
            axes[profile_index].plot(profile_feature_data)
            axes[profile_index].set_title(f'Profile {profile_index + 1}')
            axes[profile_index].set_xlabel('Time Step')
            axes[profile_index].set_ylabel('Feature Value')
            axes[profile_index].grid(True)

        # Hide any unused subplots
        for j in range(profile_index + 1, len(axes)):
            axes[j].set_visible(False)

        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.show()

train_tensor, train_sample_weights = generate_tensor(train_profiles)
test_tensor, test_sample_weights = generate_tensor(test_profiles)

# train_tensor.shape:   [43971, 66, 14]
# input_cols:           ['u_q', 'coolant', 'u_d', 'motor_speed', 'i_d', 'i_q', 'ambient', 'torque', 'i_s', 'u_s'], 10
# temperature_cols      ['pm', 'stator_yoke', 'stator_tooth', 'stator_winding', 'ambient', 'coolant']
# target_cols           ['pm', 'stator_yoke', 'stator_tooth', 'stator_winding']

# some custom activation function
class Biased_Elu(nn.Module):
    def __init__(self):
        super().__init__()
        self.elu = nn.ELU()

    def forward(self, x):
        return self.elu(x) + 1
    
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

        self.conductance_net = nn.Sequential(nn.Linear(len(input_cols) + self.output_size, n_conds),
                                             nn.Sigmoid())
        
        # self.conductance_net = nn.Sequential(
        #     nn.Linear(len(input_cols) + self.output_size, 2),
        #     nn.Tanh(),
        #     nn.Linear(2, n_conds),
        #     Biased_Elu()
        # )



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

        self.ploss = nn.Sequential(nn.Linear(len(input_cols) + self.output_size, 8),
                                    nn.Tanh(),
                                    nn.Linear(8, self.output_size),
                                   )

        # # power losses
        # self.ploss = nn.Sequential(
        #     nn.Linear(len(input_cols) + self.output_size, 4),
        #     nn.Tanh(),
        #     nn.Linear(4, self.output_size),
        #     nn.Sigmoid()
        # )

        
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
        # out = prev_out + self.sample_time * torch.exp(self.caps) * (temp_diffs + power_loss)
        out = prev_out + self.sample_time * torch.pow(10, self.caps) * (temp_diffs + power_loss)
        
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

train = True

if train:

    if not os.path.exists('outputs'):
        os.makedirs('outputs')

    # Your model definition, presumably a differential equation layer with a specific cell type.
    # torch.jit.script is used for optimizing the model for faster execution.
    model = torch.jit.script(DiffEqLayer(TNNCell).to(device))

    # In loss function parameter reduction="none" tells pytorch not to calculate average loss but leave as it is.
    # Used for weighted loss. The shape of the loss is the same as output shape!
    loss_func = nn.MSELoss(reduction="none")
    opt = optim.Adam(model.parameters(), lr=2e-3)
    n_epochs = 100
    tbptt_size = 512

    # Calculating the number of batches based on the size of your training data and the TBPTT size.
    n_batches = np.ceil(train_tensor.shape[0] / tbptt_size).astype(int)

    # Initialize the variable to keep track of the lowest loss.
    lowest_loss = float('inf')

    # Initialize a list to store loss values
    epoch_losses = []

    # Enable interactive mode for matplotlib
    plt.ion()

    # Set up the plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title('Loss Curve')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    line, = ax.plot(epoch_losses, label='Training Loss')
    ax.legend()
    ax.set_yscale('log')

    # tqdm is used to display a progress bar for your training loop.
    with tqdm(desc="Training", total=n_epochs) as pbar:
        for epoch in range(n_epochs):

            if epoch == 0:
                mdl_info = ti_summary(model, device=device, verbose=0)
                pbar.write(str(mdl_info))

            # Initial hidden state setup for the RNN, using the first data point.
            hidden = train_tensor[0, :,  -len(target_cols):]

            # Batch-wise propagation through the dataset.
            for i in range(n_batches):
                # Zero the gradients before a new step of optimization.
                model.zero_grad()

                # Forward pass: compute the model output for the current batch.
                output, hidden = model(train_tensor[i*tbptt_size:(i+1)*tbptt_size, :, :len(input_cols)], hidden.detach())

                # Calculate loss. It has the same shape as output because reduction="none".
                loss = loss_func(output, train_tensor[i*tbptt_size:(i+1)*tbptt_size, :, -len(target_cols):])

                # loss:  torch.Size([128, 66, 4])
                # weights:  torch.Size([43971, 66])
                # output:  torch.Size([128, 66, 4])


                # In weights[i*tbptt_size:(i+1)*tbptt_size, :, None] None is used to add an extra dimension at the end,
                # effectively reshaping the weights tensor from [128, 66] to [128, 66, 1].
                # Using broadcasting matrices are multiplied elementwise
                loss = (loss * train_sample_weights[i*tbptt_size:(i+1)*tbptt_size, :, None] / train_sample_weights[i*tbptt_size:(i+1)*tbptt_size, :].sum()).sum().mean()

                # Backward pass: compute gradient of the loss with respect to model parameters.
                loss.backward()
                # Perform a single optimization step (parameter update).
                opt.step()

            # Checkpointing: Save the model if the current loss is the lowest.
            current_loss = loss.item()
            if current_loss < lowest_loss:
                lowest_loss = current_loss
                torch.save(model.state_dict(), 'outputs/model_with_lowest_loss.pth')
                
            # Append the loss of the current epoch to the list
            epoch_losses.append(loss.item())

            # Update the plot
            line.set_xdata(np.arange(len(epoch_losses)))
            line.set_ydata(epoch_losses)
            ax.relim()  # Recalculate limits
            ax.autoscale_view(True, True, True)  # Rescale the view
            plt.pause(0.1)  # Pause to update the plot
            plt.savefig(os.path.join('outputs', 'loss_plot.png'))


            # Reduce learning rate after a certain number of epochs.
            if epoch == 75:
                for group in opt.param_groups:
                    group["lr"] *= 0.5

            # Update the progress bar.
            pbar.update()
            pbar.set_postfix_str(f'loss: {current_loss:.2e}')

else:
    # Load the saved model state
    model_path = '.\outputs\model_with_lowest_loss.pth'  # Replace with your model's file path
    model = torch.jit.script(DiffEqLayer(TNNCell).to(device))  # Create a model instance
    model.load_state_dict(torch.load(model_path))  # Load the trained weights
    model.eval()  # Set the model to evaluation mode

with torch.no_grad():
    pred, hidden = model(test_tensor[:, :, :len(input_cols)], test_tensor[0, :,  -len(target_cols):])
    pred = pred.cpu().numpy() * 200  # denormalize

fig, axes = plt.subplots(len(test_profiles), len(target_cols), figsize=(20, 10))
for i, (pid, y_test) in enumerate(data.loc[data.profile_id.isin(test_profiles), target_cols + ['profile_id']].groupby('profile_id')):
    y_test *= 200
    profile_pred = pred[:len(y_test), i, :]
    for j, col in enumerate(target_cols):
        ax = axes[i, j]
        ax.plot(y_test.loc[:, col].reset_index(drop=True), color='tab:green', label='Ground truth')
        ax.plot(profile_pred[:, j], color='tab:blue', label='Prediction')
        ax.text(x=0.5, y=0.8, 
                s=f'MSE: {((profile_pred[:, j] - y_test.loc[:, col])**2).sum() / len(profile_pred):.3f} K²\nmax.abs.:{(profile_pred[:, j]-y_test.loc[:, col]).abs().max():.1f} K',
                transform=ax.transAxes)
        if j == 0:
            ax.set_ylabel(f'Profile {pid}\n Temp. in °C')
            if i == 0:
                ax.legend()
        if i == len(test_profiles) - 1:
            ax.set_xlabel(f'Iters')
        elif i == 0:
            ax.set_title(col)

# Save the figure to a file
plt.savefig(os.path.join('outputs', 'test_data.png'))
plt.show()

## Just for testing!!!

# tbptt_size = 1
# n_epochs = 100
# i = 0  # For the first batch, for example
# inp = train_tensor[i * tbptt_size : (i + 1) * tbptt_size, :, :len(input_cols)]  # Selecting all features for the batch
# hidden = train_tensor[0, :,  -len(target_cols):]
# # Initialize TNNCell
# tnn_cell = TNNCell()
# # Initialize DiffEqLayer with TNNCell
# diff_eq_layer = DiffEqLayer(tnn_cell)
# output, new_hidden_state = diff_eq_layer(inp, hidden)


