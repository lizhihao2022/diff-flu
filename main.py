import logging
from time import time
from config import parser
from utils import set_up_logger, set_seed, set_device
from neuralop.models import TFNO
from training import FNOTrainer
from neuralop import LpLoss
from dataset import KMFlowDataset
import torch


def main():
    # initial
    args = parser.parse_args()
    saving_path, saving_name = set_up_logger(args)
    set_device(args.cuda, args.device)
    set_seed(args.random_seed)
    
    # load data
    logging.info("Loading {} dataset".format(args.dataset))
    start = time()
    dataset = KMFlowDataset(data_dir=args.data_dir, batch_size=args.batch_size)
    train_loader, valid_loader, test_loader = dataset.train_loader, dataset.valid_loader, dataset.test_loader
    logging.info("Loading data costs {: .2f}s".format(time() - start))
    
    # build model
    logging.info("Building models")
    start = time()
    model = TFNO(
        n_modes=(256, 256),
        in_channels=1,
        out_channels=1,
        n_layers=args.layers,
        hidden_channels=32, 
        projection_channels=64, 
        factorization='tucker', 
        rank=0.42
    )
    model = model.to('cuda')                                                   
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    criterion = LpLoss(d=2, p=2)
    eval_metrics = ['l2', 'h1']
    logging.info("Model: {}".format(model))
    logging.info("Optimizer: {}".format(optimizer))
    logging.info("Scheduler: {}".format(scheduler))
    logging.info("Criterion: {}".format(criterion))
    logging.info("Eval metrics: {}".format(eval_metrics))
    logging.info("Building models costs {: .2f}s".format(time() - start))
    
    trainer = FNOTrainer(model, epochs=args.epochs,
                         device='cuda',
                         eval_freq=args.eval_freq,
                         patience=args.patience,
                         verbose=args.verbose,
                         logger=args.log, 
                         wandb_log=args.wandb, 
                         mg_patching_levels=0, 
                         use_distributed=False,
                         saving_path=saving_path,)
    
    trainer.process(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        eval_metrics=eval_metrics,
    )


if __name__ == "__main__":
    main()
