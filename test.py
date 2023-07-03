from utils import set_seed, set_device, train, evaluate
from dataset import KMFlowDataset
from config import parser
import torch
from neuralop.models import TFNO
from neuralop import LpLoss, H1Loss
import sys
from neuralop import Trainer
import torch.distributed as dist
import os

os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '29500'

dist.init_process_group(backend='nccl', rank=0, world_size=1)

device = 'cuda'

args = parser.parse_args()
args.epochs = 100
args.subset = True

# set_device(args.cuda, args.device)
set_seed(args.random_seed)

dataset = KMFlowDataset(data_dir="./data/km_flow/", batch_size=128)
train_loader, valid_loader, test_loader = dataset.train_loader, dataset.valid_loader, dataset.test_loader    

model = TFNO(
    n_modes=(256, 256),
    in_channels=1,
    out_channels=1,
    hidden_channels=32, 
    projection_channels=64, 
    factorization='tucker', 
    rank=0.42
)
model = model.to('cuda')

optimizer = torch.optim.Adam(model.parameters(), lr=8e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

# Creating the losses
l2loss = LpLoss(d=2, p=2)
h1loss = H1Loss(d=2)

train_loss = h1loss
eval_losses={'h1': h1loss, 'l2': l2loss}

print('\n### MODEL ###\n', model)
print('\n### OPTIMIZER ###\n', optimizer)
print('\n### SCHEDULER ###\n', scheduler)
print('\n### LOSSES ###')
print(f'\n * Train: {train_loss}')
print(f'\n * Test: {eval_losses}')
sys.stdout.flush()

# Create the trainer
trainer = Trainer(model, n_epochs=20,
                  device=device,
                  mg_patching_levels=0,
                  wandb_log=False,
                  log_test_interval=3,
                  use_distributed=True,
                  verbose=True)


# Actually train the model on our small Darcy-Flow dataset

trainer.train(train_loader, test_loader,
              None,
              model, 
              optimizer,
              scheduler, 
              regularizer=False, 
              training_loss=train_loss,
              eval_losses=eval_losses)
