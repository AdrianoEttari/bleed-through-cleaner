from tqdm import tqdm
import torch
import os
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler
from utils import get_data
from torch.utils.data import DataLoader
from UNet_model import Residual_Attention_UNet

class Trainer:
    def __init__(
            self,
            multiple_gpus: bool,
            save_every: int,
            model: torch.nn.Module,
            snapshot_path,
            train_data,
            optimizer,
            device,
            ) -> None:
        
        self.multiple_gpus = multiple_gpus
        self.save_every = save_every
        # self.loss_function = torch.nn.MSELoss()
        self.loss_function = torch.nn.BCELoss()
        self.device = device
        if self.multiple_gpus:
            self.model = DDP(model, device_ids=[self.device], find_unused_parameters=True)
        else:
            self.model = model.to(self.device)

        self.model.train()
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.epochs_run = 0
        self.snapshot_path = snapshot_path

        if not os.path.exists(self.snapshot_path):
            os.makedirs(self.snapshot_path, exist_ok=True)
            print(f"Checkpoint folder '{self.snapshot_path}' created successfully.")

        if os.path.exists(os.path.join(self.snapshot_path, "snapshot.pt")):
            print(f"Loading snapshot from {self.snapshot_path}")
            self.load_snapshot(os.path.join(self.snapshot_path, "snapshot.pt"))   

    def run_epoch(self, epoch: int):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"\n\n[GPU{self.device}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        if self.multiple_gpus:
            self.train_data.sampler.set_epoch(epoch)

        running_train_loss = 0.0
        for idx, (source, targets) in tqdm(enumerate(self.train_data), total=len(self.train_data)):
            source, targets = source.to(self.device), targets.to(self.device)
            loss = self.run_batch(source, targets)
            running_train_loss += loss

        running_train_loss /= len(self.train_data)
        print(f"Epoch: {epoch} | Training Loss: {running_train_loss}")

    def run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = self.loss_function(output, targets)
        loss.backward()
        self.optimizer.step()
        return loss.item()  

    def save_snapshot(self, epoch: int, snapshot_path: str):
        snapshot = {}
        if self.multiple_gpus:
            snapshot["MODEL_STATE"] = self.model.module.state_dict() # model.module is the state dict of the model wrapped by DDP
        else:
            snapshot["MODEL_STATE"] = self.model.state_dict()
        snapshot["EPOCHS_RUN"] = epoch
        PATH = os.path.join(snapshot_path, "snapshot.pt")
        torch.save(snapshot, PATH)
        print(f"Epoch {epoch} | Training snapshot saved at {PATH}")

    def load_snapshot(self, snapshot_path: str):
        snapshot = torch.load(snapshot_path,map_location=torch.device('cpu'), weights_only=True)
        if self.multiple_gpus:
            self.model.module.load_state_dict(snapshot["MODEL_STATE"])
            self.model.module.to(self.device)
        else:
            self.model.load_state_dict(snapshot["MODEL_STATE"])
            self.model.to(self.device)
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Snapshot loaded from {snapshot_path}")

    def train(self, max_epochs: int, snapshot_path: str, scheduler):
        if os.path.exists(os.path.join(snapshot_path, "snapshot.pt")):
            snapshot = torch.load(os.path.join(snapshot_path, 'snapshot.pt'),map_location=torch.device('cpu'), weights_only=True)
            print(f"Restart training from epoch {snapshot['EPOCHS_RUN']}")
            
        for epoch in tqdm(range(self.epochs_run + 1, max_epochs +1), desc="Training the network"):
            self.run_epoch(epoch)

            if scheduler:
                scheduler.step()
                print(f"Learning rate: {self.optimizer.param_groups[0]['lr']}")

            # Save checkpoint from the rank 0 process
            if self.multiple_gpus:
                if self.device == 0 and epoch % self.save_every == 0:
                    self.save_snapshot(epoch, snapshot_path)
            else:
                if epoch % self.save_every == 0:
                    self.save_snapshot(epoch, snapshot_path)


def prepare_data_loader(dataset_path, _set, multiple_gpus, batch_size):
    full_dataset_path = os.path.join(dataset_path, _set)
    dataset = get_data(full_dataset_path, "pages", "masks")

    if multiple_gpus:
        data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, sampler=DistributedSampler(dataset, shuffle=True))
    else:
        data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    
    return data_loader
    
def launch(args):
    multiple_gpus = args.multiple_gpus
    dataset_path = args.dataset_path
    snapshot_folder_path = args.snapshot_folder_path
    model_name = args.model_name
    batch_size = args.batch_size
    lr = args.lr
    num_epochs = args.num_epochs
    save_every = args.save_every
    out_dim = args.out_dim
    input_channels = args.input_channels

    os.makedirs(snapshot_folder_path, exist_ok=True)
    snapshot_path = os.path.join(snapshot_folder_path, model_name)

    if multiple_gpus:
        print("Using multiple GPUs")
        init_process_group(backend="nccl")
        device = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(int(device))
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # device = 'mps'
        print(f"Using device: {device}")

    train_loader = prepare_data_loader(dataset_path, "train", multiple_gpus, batch_size)

    model = Residual_Attention_UNet(image_channels=input_channels, out_dim=out_dim, device=device).to(device)
    print("Num params: ", sum(p.numel() for p in model.parameters()))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    trainer = Trainer(multiple_gpus, save_every, model, snapshot_path, train_loader, optimizer, device)
    trainer.train(num_epochs, snapshot_path, scheduler=None)

    if multiple_gpus:
        destroy_process_group()


if __name__ == "__main__":
    import argparse     

    def str2bool(v):
        """Convert string to boolean."""
        return v.lower() in ("yes", "true", "t", "1")
    
    parser = argparse.ArgumentParser(description="Training script for the UNet model")
    parser.add_argument("--multiple_gpus", type=str2bool, default=False, help="Use multiple GPUs")
    parser.add_argument("--save_every", type=int, default=5, help="Save snapshot every n epochs")
    parser.add_argument("--snapshot_folder_path", type=str, help="Folder path to save snapshots")
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset")
    parser.add_argument("--model_name", type=str, help="Name of the model")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--out_dim", type=int, default=1, help="Output dimension of the model")
    parser.add_argument("--input_channels", type=int, default=3, help="Number of input channels")


    args = parser.parse_args()
    launch(args)

