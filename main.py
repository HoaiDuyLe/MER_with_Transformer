import os
import time
# from turtle import shape
import torch
import numpy as np
from torch.utils.data import DataLoader
from src.cli import get_args
from src.datasets import get_dataset_iemocap, collate_fn, get_dataset_mosei
from src.models.e2e import MME2E
from src.trainers.emotiontrainer import IemocapTrainer

if __name__ == "__main__":
    start = time.time()

    args = get_args()

    # Fix seed for reproducibility
    seed = args['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set device
    os.environ["CUDA_VISIBLE_DEVICES"] = args['cuda']
    device = torch.device(f"cuda:{args['cuda']}" if torch.cuda.is_available() else 'cpu')

    print("Start loading the data....")

    if args['dataset'] == 'iemocap':
        train_dataset = get_dataset_iemocap(data_folder=args['datapath'], phase='train',
                                            img_interval=args['img_interval'], hand_crafted_features=args['hand_crafted'])
        valid_dataset = get_dataset_iemocap(data_folder=args['datapath'], phase='valid',
                                            img_interval=args['img_interval'], hand_crafted_features=args['hand_crafted'])
        test_dataset = get_dataset_iemocap(data_folder=args['datapath'], phase='test',
                                           img_interval=args['img_interval'], hand_crafted_features=args['hand_crafted'])

    elif args['dataset'] == 'mosei':
        train_dataset = get_dataset_mosei(data_folder=args['datapath'], phase='train', img_interval=args['img_interval'])
        valid_dataset = get_dataset_mosei(data_folder=args['datapath'], phase='valid', img_interval=args['img_interval'])
        test_dataset = get_dataset_mosei(data_folder=args['datapath'], phase='test', img_interval=args['img_interval'])

    train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True, num_workers=4, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=args['batch_size'], shuffle=False, num_workers=4, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=False, num_workers=4, collate_fn=collate_fn)

    print(f'# Train samples = {len(train_loader.dataset)}')
    print(f'# Valid samples = {len(valid_loader.dataset)}')
    print(f'# Test samples = {len(test_loader.dataset)}')

    dataloaders = {
        'train': train_loader,
        'valid': valid_loader,
        'test': test_loader
    }

    lr = args['learning_rate']
    if args['model'] == 'mme2e':
        model = MME2E(args=args, device=device)
        model = model.to(device=device)

        # When using a pre-trained text modal, you can use text_lr_factor to give a smaller leraning rate to the textual model parts
        if args['text_lr_factor'] == 1:
            optimizer = torch.optim.Adam(model.parameters(), lr=args['learning_rate'], weight_decay=args['weight_decay'])
        else:
            optimizer = torch.optim.Adam([
                {'params': model.T.parameters(), 'lr': lr / args['text_lr_factor']},
                {'params': model.V.parameters()},
                {'params': model.v_flatten.parameters()},
                {'params': model.v_transformer.parameters()},
                {'params': model.A.parameters()},
                {'params': model.a_flatten.parameters()},
                {'params': model.a_transformer.parameters()}, 
                {'params': model.f_transformer.parameters()},
                # {'params': model.out.parameters()},  
                {'params': model.f_decoder.parameters()},   
                {'params': model.fc.parameters()},                            
            ], lr=lr, weight_decay=args['weight_decay'])
    else:
        raise ValueError('Incorrect model name!')

    if args['scheduler']:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args['epochs'] * len(train_loader.dataset) // args['batch_size'])
    else:
        scheduler = None

    if args['loss'] == 'ce':
        criterion = torch.nn.CrossEntropyLoss()
    elif args['loss'] == 'bce':
        pos_weight = train_dataset.getPosWeight()
        pos_weight = torch.tensor(pos_weight).to(device)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    if args['dataset'] == 'iemocap' or 'mosei':
        trainer = IemocapTrainer(args, model, criterion, optimizer, scheduler, device, dataloaders)
    else:
        print("Incorrect dataset name!!")

    if args['test']:
        trainer.test()
    elif args['valid']:
        trainer.valid()
    else:
        trainer.train()

    print(f'Total time usage = {(time.time() - start) / 3600:.2f} hours.')
