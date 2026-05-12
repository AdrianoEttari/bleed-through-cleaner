#%%
import torch
import numpy as np
import os
from tqdm import tqdm
import torch
import os
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler
from utils_inpainting import get_data

class masked_l1(torch.nn.Module):
    def __init__(self):
        super(masked_l1, self).__init__()

    def forward(self, pred, gt, mask):
        return torch.mean(
            torch.abs(pred - gt) * mask.unsqueeze(1)
        )
class Trainer:
    def __init__(
            self,
            multiple_gpus: bool,
            save_every: int,
            model: torch.nn.Module,
            snapshot_path,
            snapshot_filename,
            train_data,
            optimizer,
            device,
            ) -> None:
        
        self.multiple_gpus = multiple_gpus
        self.save_every = save_every

        self.device = device
        self.loss_function = masked_l1()
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
        self.snapshot_filename = snapshot_filename
        
        if not os.path.exists(self.snapshot_path):
            os.makedirs(self.snapshot_path, exist_ok=True)
            print(f"Checkpoint folder '{self.snapshot_path}' created successfully.")

        if os.path.exists(os.path.join(self.snapshot_path, self.snapshot_filename)):
            print(f"Loading snapshot from {self.snapshot_path}")
            self.load_snapshot(os.path.join(self.snapshot_path, self.snapshot_filename))   

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
        rgb_corrupt = source[:, :3, :, :]
        text_mask = source[:, 3, :, :]
        holes_mask = source[:, 4, :, :]
        output = self.model(rgb_corrupt)
        mask = text_mask * holes_mask
        loss = self.loss_function(output, targets, mask)
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
        PATH = os.path.join(snapshot_path, self.snapshot_filename)
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
        if os.path.exists(os.path.join(snapshot_path, self.snapshot_filename)):
            snapshot = torch.load(os.path.join(snapshot_path, self.snapshot_filename),map_location=torch.device('cpu'), weights_only=True)
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

#%%
from UNet_model_inpainting import ResidualUNet

patch_size = 1024

multiple_gpus=False
save_every=1
batch_size=16
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("USING", device, " device")
snapshot_path="./snapshots"
os.makedirs(snapshot_path, exist_ok=True)

snapshot_filename = "snapshot_PathSize_"+str(patch_size)+".pt"

dataset_path = os.path.join("dataset_MAGIC_PatchSize"+str(patch_size))
# dataset_path = os.path.join("/data1","aettari","dataset_MAGIC_PatchSize"+str(patch_size))

train_dataset = get_data(".", dataset_path, max_hole_size=300)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
model=ResidualUNet(in_channels=3, out_channels=3, channels=(32, 64, 128, 256), device=device).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
num_epochs=500

trainer = Trainer(multiple_gpus, save_every, model, snapshot_path, snapshot_filename, train_loader, optimizer, device)
trainer.train(num_epochs, snapshot_path, scheduler=None)

# %%
