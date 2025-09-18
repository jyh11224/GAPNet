# import argparse
# import time

# import torch.optim as optim

# from geotransformer.engine import EpochBasedTrainer

# from config import make_cfg
# from dataset import train_valid_data_loader
# from model import create_model
# from loss import OverallLoss, Evaluator


# class Trainer(EpochBasedTrainer):
#     def __init__(self, cfg):
#         super().__init__(cfg, max_epoch=cfg.optim.max_epoch)

#         # dataloader
#         start_time = time.time()
#         train_loader, val_loader, neighbor_limits = train_valid_data_loader(cfg, self.distributed)
#         loading_time = time.time() - start_time
#         message = 'Data loader created: {:.3f}s collapsed.'.format(loading_time)
#         self.logger.info(message)
#         message = 'Calibrate neighbors: {}.'.format(neighbor_limits)
#         self.logger.info(message)
#         self.register_loader(train_loader, val_loader)

#         # model, optimizer, scheduler
#         model = create_model(cfg).cuda()
#         model = self.register_model(model)
#         optimizer = optim.Adam(model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
#         self.register_optimizer(optimizer)
#         scheduler = optim.lr_scheduler.StepLR(optimizer, cfg.optim.lr_decay_steps, gamma=cfg.optim.lr_decay)
#         self.register_scheduler(scheduler)

#         # loss function, evaluator
#         self.loss_func = OverallLoss(cfg).cuda()
#         self.evaluator = Evaluator(cfg).cuda()

#     def train_step(self, epoch, iteration, data_dict):
#         output_dict = self.model(data_dict)
#         loss_dict = self.loss_func(output_dict, data_dict)
#         result_dict = self.evaluator(output_dict, data_dict)
#         loss_dict.update(result_dict)
#         return output_dict, loss_dict

#     def val_step(self, epoch, iteration, data_dict):
#         output_dict = self.model(data_dict)
#         loss_dict = self.loss_func(output_dict, data_dict)
#         result_dict = self.evaluator(output_dict, data_dict)
#         loss_dict.update(result_dict)
#         return output_dict, loss_dict


# def main():
#     cfg = make_cfg()
#     trainer = Trainer(cfg)
#     trainer.run()


# if __name__ == '__main__':
#     main()
import argparse
import time
import torch
import torch.optim as optim

from geotransformer.engine import EpochBasedTrainer

from config_refine import make_cfg
from dataset import train_valid_data_loader
from model import create_model
from loss import OverallLoss, Evaluator


class Trainer(EpochBasedTrainer):
    def __init__(self, cfg):
        super().__init__(cfg, max_epoch=cfg.optim.max_epoch)

        # dataloader
        start_time = time.time()
        train_loader, val_loader, neighbor_limits = train_valid_data_loader(cfg, self.distributed)
        loading_time = time.time() - start_time
        self.logger.info(f'Data loader created: {loading_time:.3f}s.')
        self.logger.info(f'Calibrate neighbors: {neighbor_limits}.')
        self.register_loader(train_loader, val_loader)

        # model
        model = create_model(cfg).cuda()

        # === Pretrained load (optional) ===
        pretrain_path = getattr(cfg.model, "pretrain_path", "/workspace/GeoTransformer/geotransformer-3dmatch.pth.tar")
        if pretrain_path:
            state = torch.load(pretrain_path, map_location="cpu")
            if "state_dict" in state:
                state = state["state_dict"]
            missing, unexpected = model.load_state_dict(state, strict=False)
            self.logger.info(f"[Pretrain] loaded from: {pretrain_path}")
            self.logger.info(f"[Pretrain] missing keys: {len(missing)} | unexpected keys: {len(unexpected)}")

        # register (handles DDP)
        model = self.register_model(model)

        # === Optimizer & Scheduler: 전체 파인튜닝 ===
        optimizer = optim.Adam(model.parameters(), lr=cfg.model.lr, weight_decay=cfg.optim.weight_decay)
        self.register_optimizer(optimizer)
        scheduler = optim.lr_scheduler.StepLR(optimizer, cfg.optim.lr_decay_steps, gamma=cfg.optim.lr_decay)
        self.register_scheduler(scheduler)

        # losses & evaluator
        self.loss_func = OverallLoss(cfg).cuda()
        self.evaluator = Evaluator(cfg).cuda()

    def train_step(self, epoch, iteration, data_dict):
        output_dict = self.model(data_dict)
        loss_dict = self.loss_func(output_dict, data_dict)
        result_dict = self.evaluator(output_dict, data_dict)
        loss_dict.update(result_dict)
        return output_dict, loss_dict

    def val_step(self, epoch, iteration, data_dict):
        output_dict = self.model(data_dict)
        loss_dict = self.loss_func(output_dict, data_dict)
        result_dict = self.evaluator(output_dict, data_dict)
        loss_dict.update(result_dict)
        return output_dict, loss_dict


def main():
    cfg = make_cfg()

    # 🔧 중요 플래그 예시 (config 내부에서 세팅 권장)
    cfg.model.glue_refine = True
    cfg.model.glue_detach = False         # 전체 학습이면 False 추천
    cfg.model.refiner_num_heads = 1
    cfg.model.refiner_n_layers = 4
    cfg.model.refiner_descriptor_dim = 64
    cfg.model.descriptor_dim = 256        # Refiner 미사용 시
    cfg.model.pretrain_path = "/workspace/GeoTransformer/geotransformer-3dmatch.pth.tar"

    trainer = Trainer(cfg)
    trainer.run()


if __name__ == '__main__':
    main()
