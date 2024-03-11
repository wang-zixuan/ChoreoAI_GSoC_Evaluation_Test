import torch
from functions.load_data import * 
from torch_geometric.data import Data

class MarielDataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, reduced_joints=False, xy_centering=True, seq_len=128, predicted_timesteps=1, file_path="data/mariel_*.npy", no_overlap=False, use_graph=False):
        'Initialization'
        self.file_path      = file_path
        self.seq_len        = seq_len
        self.no_overlap     = no_overlap
        self.reduced_joints = reduced_joints # use a meaningful subset of joints
        self.data           = load_data(pattern=file_path) 
        self.xy_centering   = xy_centering
        self.n_joints       = 53
        self.n_dim          = 6
        self.predicted_timesteps = predicted_timesteps
        self.use_graph = use_graph
        
        print("")
        
        if self.no_overlap:
            print("Generating non-overlapping sequences...")   
        else:
            print("Generating overlapping sequences...")
        
        if self.xy_centering:
            print("Using (x, y)-centering...")
        else:
            print("Not using (x, y)-centering...")
            
        if self.reduced_joints: 
            print("Reducing joints...")
        else:
            print("Using all joints...")

    def __len__(self):
        'Denotes the total number of samples'
        if self.xy_centering: 
            data = self.data[1] # choose index 1, for the (x,y)-centered phrases
        else: 
            data = self.data[0] # choose index 0, for data without (x,y)-centering
        
        if self.no_overlap:
             # number of complete non-overlapping phrases
            return int(len(data) / self.seq_len)
        else:
            # number of overlapping phrases up until the final complete phrase
            return len(data) - self.seq_len 

    def __getitem__(self, index):
        'Generates one sample of data'
        edge_index, is_skeleton_edge, reduced_joint_indices = edges(reduced_joints=self.reduced_joints, seq_len=self.seq_len)
        
        if self.xy_centering:
            data = self.data[1] # choose index 1, for the (x,y)-centered phrases
        else:
            data = self.data[0] # choose index 0, for data without (x,y)-centering

        if self.reduced_joints:
            data = data[:, reduced_joint_indices, :] # reduce number of joints if desired

        if self.no_overlap:
            # non-overlapping phrases
            index = index * self.seq_len
            sequence = data[index: index + self.seq_len]
            # prediction_target = data[index: index + self.seq_len + self.predicted_timesteps]
        else:
            # overlapping phrases
            sequence = data[index: index + self.seq_len]
            # prediction_target = data[index: index + self.seq_len + self.predicted_timesteps]

        # [seq_len, joints, 6]
        if not self.use_graph:
            cur_data = {}
            cur_data['seq'] = torch.Tensor(sequence)
            # cur_data['target'] = torch.Tensor(prediction_target)
            return cur_data

        sequence = np.transpose(sequence, [1, 0, 2]) # put n_joints first
        sequence = sequence.reshape((data.shape[1], self.n_dim * self.seq_len)) # flatten n_dim * seq_len into one dimension (i.e. node feature)
        prediction_target = np.transpose(prediction_target, [1, 0, 2]) # put n_joints first
        prediction_target = prediction_target.reshape((data.shape[1], self.n_dim * (self.seq_len + self.predicted_timesteps))) 

        # Convert to torch objects
        sequence = torch.Tensor(sequence)
        prediction_target = torch.Tensor(prediction_target)
        edge_attr = torch.Tensor(is_skeleton_edge)

        return Data(x=sequence, y=prediction_target, edge_index=edge_index.t().contiguous(), edge_attr=edge_attr)
