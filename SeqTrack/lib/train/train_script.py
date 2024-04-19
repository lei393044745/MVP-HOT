import os
# loss function related
from lib.utils.box_ops import giou_loss
from torch.nn.functional import l1_loss
from torch.nn import BCEWithLogitsLoss, MSELoss, CrossEntropyLoss
# train pipeline related
from lib.train.trainers import LTRTrainer
# distributed training related
from torch.nn.parallel import DistributedDataParallel as DDP
# some more advanced functions
from .base_functions import *
# network related
from lib.models.seqtrack import build_seqtrack
# forward propagation related
from lib.train.actors import SeqTrackActor
# for import modules
import importlib
import torch.nn as nn

def run(settings):
    settings.description = 'Training script for SeqTrack'

    # update the default configs with config file
    if not os.path.exists(settings.cfg_file):
        raise ValueError("%s doesn't exist." % settings.cfg_file)
    config_module = importlib.import_module("lib.config.%s.config" % settings.script_name)
    cfg = config_module.cfg # generate cfg from lib.config
    config_module.update_config_from_file(settings.cfg_file) #update cfg from experiments
    if settings.local_rank in [-1, 0]:
        print("New configuration is shown below.")
        for key in cfg.keys():
            print("%s configuration:" % key, cfg[key])
            print('\n')

    # update settings based on cfg
    update_settings(settings, cfg)

    # Record the training log
    log_dir = os.path.join(settings.save_dir, 'logs')
    if settings.local_rank in [-1, 0]:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    settings.log_file = os.path.join(log_dir, "%s-%s.log" % (settings.script_name, settings.config_name))

    # Build dataloaders
    loader_type = getattr(cfg.DATA, "LOADER", "tracking")
    if loader_type == "tracking":
        loader_train = build_dataloaders(cfg, settings)
    else:
        raise ValueError("illegal DATA LOADER")


    # Create network
    if settings.script_name == "seqtrack":
        net = build_seqtrack(cfg)        #pix2seq method with multi-frames and encoder mask
    else:
        raise ValueError("illegal script name")

    # wrap networks to distributed one
    net.cuda()
    weight_path = '/media/fox/6E3464883464555D2/xsx/VideoX-master/SeqTrack/checkpoints/SEQTRACK_ep0500.pth.tar'
    # weight_path = '/media/fox/6E3464883464555D/xsx/VideoX-master/SeqTrack/checkpoints/train/seqtrack/seqtrack_b256/SEQTRACK_ep0001.pth.tar'
    state_dict = torch.load(weight_path)
    net.load_state_dict(state_dict['net'], strict=False)
    nets = [net.encoder.body.prompt, net.encoder.body.hsi_patch_embed]
    
    for param in net.parameters():
            param.requires_grad = False
    for m in net.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
            m.eval()
    for train_net in nets:
        for param in train_net.parameters():
            param.requires_grad = True
        for m in train_net.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                m.train()
    net.encoder.body.hsi_pos_embed.requires_grad = True
    # net.encoder.body.lamdas.requires_grad = True
    n_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    n_parameters = sum(p.numel() for p in net.parameters() if not p.requires_grad)
    print('number of params:', n_parameters)
    if settings.local_rank != -1:
        net = DDP(net, broadcast_buffers=False, device_ids=[settings.local_rank], find_unused_parameters=False)
        settings.device = torch.device("cuda:%d" % settings.local_rank)
    else:
        settings.device = torch.device("cuda:0")
    settings.deep_sup = getattr(cfg.TRAIN, "DEEP_SUPERVISION", False)
    settings.distill = getattr(cfg.TRAIN, "DISTILL", False)
    settings.distill_loss_type = getattr(cfg.TRAIN, "DISTILL_LOSS_TYPE", "KL")
    # Loss functions and Actors
    if settings.script_name == "seqtrack":
        bins = cfg.MODEL.BINS
        weight = torch.ones(bins + 2)
        weight[bins] = 0.01
        weight[bins + 1] = 0.01
        objective = {'ce': CrossEntropyLoss(weight=weight)}
        loss_weight = {'ce': cfg.TRAIN.CE_WEIGHT}
        actor = SeqTrackActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, cfg=cfg)

    else:
        raise ValueError("illegal script name")

    # Optimizer, parameters, and learning rates
    optimizer, lr_scheduler = get_optimizer_scheduler(net, cfg)
    use_amp = getattr(cfg.TRAIN, "AMP", False)
    trainer = LTRTrainer(actor, [loader_train], optimizer, settings, lr_scheduler, use_amp=use_amp)

    # train process
    trainer.train(cfg.TRAIN.EPOCH, load_latest=True, fail_safe=True)
