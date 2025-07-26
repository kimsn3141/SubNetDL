import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.model_selection import StratifiedKFold, KFold
from datetime import datetime

from data_loader import load_data
from model import SubNetDL

def main():

    ppi_network, dataset = load_data()
    graph_mask = torch.Tensor(ppi_network)
    graph_mask.requires_grad = False
    number_of_genes = len(ppi_network)

    train_loader, val_loader, test_loader = _split_dataset(dataset)
    model_out_path = _make_output_dir()

    epoch = 10
    network_form = [(5, 10, 5), (50, 10, 3)]
    learning_rate = 1 / 10000
    device = "cuda:0"

    train_pat_number = len(train_loader)
    total_steps = epoch * train_pat_number

    model = SubNetDL(graph_mask, network_form, number_of_genes, act_func="prelu").to(torch.device(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=2, eta_min=1e-7)
    loss_fn = nn.BCEWithLogitsLoss()

    model.train()
    step = 0

    best_val_loss = float('inf')
    best_model_path = os.path.join(model_out_path, "best_model.pt")

    while step < total_steps:
        for batch in train_loader:
            inputs = batch["input"][0].to(device)
            labels = batch["response"].to(device)

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(step)

            if step % train_pat_number == 0:
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for val_batch in val_loader:
                        val_input = val_batch["input"][0].to(device)
                        val_label = val_batch["response"].to(device)
                        val_output = model(val_input)
                        val_loss += loss_fn(val_output, val_label).item()
                avg_val_loss = val_loss / len(val_loader)
                print(f"[Step {step}] Validation Loss: {avg_val_loss:.4f}")

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    torch.save(model.state_dict(), best_model_path)
                    print(f"New best model saved at step {step} with val loss {avg_val_loss:.4f}")

                model.train()

            step += 1
            if step >= total_steps:
                break

    print(f"Loading best model from: {best_model_path}")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    # Final evaluation on test set
    test_loss = 0.0
    preds, trues = [], []
    with torch.no_grad():
        for test_batch in test_loader:
            test_input = test_batch["input"][0].to(device)
            test_label = test_batch["response"].to(device)
            test_output = model(test_input)
            test_loss += loss_fn(test_output, test_label).item()

            preds.append(torch.sigmoid(test_output).cpu())
            trues.append(test_label.cpu())

    avg_test_loss = test_loss / len(test_loader)
    print(f"[Test] Average Loss: {avg_test_loss:.4f}")

def _split_dataset(dataset, seed = 42):
    responses = np.array([int(sample['response'].item()) for sample in dataset])

    skf1 = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    train_idx, temp_idx = next(skf1.split(np.zeros(len(responses)), responses))

    temp_responses = responses[temp_idx]
    unique_labels, label_counts = np.unique(temp_responses, return_counts=True)

    if len(unique_labels) >= 2 and np.all(label_counts >= 2):
        skf2 = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
        val_idx_local, test_idx_local = next(skf2.split(np.zeros(len(temp_responses)), temp_responses))
    else:
        kf = KFold(n_splits=2, shuffle=True, random_state=seed)
        val_idx_local, test_idx_local = next(kf.split(np.zeros(len(temp_responses))))

    val_idx = temp_idx[val_idx_local]
    test_idx = temp_idx[test_idx_local]

    # Subsets
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    test_subset = Subset(dataset, test_idx)

    # Dataloaders
    sampler = create_balanced_sampler(train_subset)
    train_loader = DataLoader(train_subset, batch_size=1, sampler=sampler)
    val_loader = DataLoader(val_subset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=1, shuffle=False)

    return train_loader, val_loader, test_loader

def _make_output_dir():
        today = datetime.today()
        now_time = today.strftime("%Y_%m_%d_%H_%M_%S")

        folder_path = f"data/result/{now_time}/"
        os.makedirs(folder_path)
        model_out_path = folder_path + "best_model/"
        os.makedirs(model_out_path)

        return model_out_path


def create_balanced_sampler(dataset):
    labels = [sample["response"].item() for sample in dataset]
    labels = np.array(labels)

    class_sample_count = np.bincount(labels.astype(int))
    weight_per_class = 1.0 / class_sample_count

    weights = weight_per_class[labels.astype(int)]

    sampler = WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)
    return sampler

if __name__ == "__main__":
    main()