import json
import torch
from torch.utils.data import Dataset
import pandas as pd
from collections import defaultdict

class PatientDataset(Dataset):
    def __init__(self, data_dict):
        self.inputs = []
        self.responses = []

        for patient_id, patient_data in data_dict.items():
            input_tensor = torch.tensor(patient_data['input'], dtype=torch.float32)
            response_tensor = torch.tensor(patient_data['response'], dtype=torch.float32)
            self.inputs.append(input_tensor)
            self.responses.append(response_tensor)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {
            'input': self.inputs[idx],
            'response': self.responses[idx]
        }


def load_data()-> tuple[list, PatientDataset]:
    """
    Load data to train SubNetDL

    Returns:
        list: adjacency matrix to express PPI network.
        PatientDataset: dataset to train SubNetDL
    """

    ppi_network = list()
    ppi_path = "data/mock_network.csv"
    ppi_df = pd.read_csv(ppi_path, index_col=["gene"])
    for i, row in ppi_df.iterrows():
        ppi_network.append([])
        for col in ppi_df.columns:
            ppi_network[-1].append(1 / (row[col] + 1))

    patient_data_path = "data/mock_pat_data.json"
    with open(patient_data_path) as f:
        data = json.load(f)

    patient_data = PatientDataset(data)

    response_counter = defaultdict(int)
    for data in patient_data:
        response_counter[data["response"].item()]+=1
    
    print(f"load network with {len(ppi_df.columns)} genes")
    print(
        (
            f"load {len(patient_data)} patients with "+
            f"{response_counter[1.0]} responder and {response_counter[0.0]} non-response"
        )
        )

    return (ppi_network, patient_data)

