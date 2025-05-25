import argparse
import comet_ml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pytorch_lightning as pl
import warmup_scheduler
import numpy as np
import torchmetrics  # 引入 torchmetrics
import pynvml

from utils import get_model, get_dataset, get_experiment_name, get_criterion
from da import CutMix, MixUp

def found_device():
    default_device = 1
    default_memory_threshold = 500
    pynvml.nvmlInit()
    while True:
        handle = pynvml.nvmlDeviceGetHandleByIndex(default_device)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        used = meminfo.used / 1024 ** 2
        if used < default_memory_threshold:
            break
        else:
            default_device += 1
        if default_device >= 8:
            default_device = 0
            default_memory_threshold += 1000
    pynvml.nvmlShutdown()
    return str(default_device)

device = 'cuda:' + found_device() if torch.cuda.is_available() else 'cpu'
print(f"[INFO] Using device: {device}")

parser = argparse.ArgumentParser()
parser.add_argument("--api-key", help="API Key for Comet.ml")
parser.add_argument("--dataset", default="c10", type=str, help="[c10, c100, svhn]")
parser.add_argument("--num-classes", default=10, type=int)
parser.add_argument("--model-name", default="vit", help="[vit]", type=str)
parser.add_argument("--patch", default=8, type=int)
parser.add_argument("--batch-size", default=128, type=int)
parser.add_argument("--eval-batch-size", default=1024, type=int)
parser.add_argument("--lr", default=1e-3, type=float)
parser.add_argument("--min-lr", default=1e-5, type=float)
parser.add_argument("--beta1", default=0.9, type=float)
parser.add_argument("--beta2", default=0.999, type=float)
parser.add_argument("--off-benchmark", action="store_true")
parser.add_argument("--max-epochs", default=200, type=int)
parser.add_argument("--dry-run", action="store_true")
parser.add_argument("--weight-decay", default=5e-5, type=float)
parser.add_argument("--warmup-epoch", default=5, type=int)
parser.add_argument("--precision", default=16, type=int)
parser.add_argument("--autoaugment", action="store_true")
parser.add_argument("--criterion", default="ce")
parser.add_argument("--label-smoothing", action="store_true")
parser.add_argument("--smoothing", default=0.1, type=float)
parser.add_argument("--rcpaste", action="store_true")
parser.add_argument("--cutmix", action="store_true")
parser.add_argument("--mixup", action="store_true")
parser.add_argument("--dropout", default=0.0, type=float)
parser.add_argument("--head", default=12, type=int)
parser.add_argument("--num-layers", default=7, type=int)
parser.add_argument("--hidden", default=384, type=int)
parser.add_argument("--mlp-hidden", default=384, type=int)
parser.add_argument("--off-cls-token", action="store_true")
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--project-name", default="VisionTransformer")
parser.add_argument("--average-number", default=100, type=int, help="Number of sample pairs for region count")
parser.add_argument("--scope-l", default=0.0, type=float, help="Left scope for interpolation")
parser.add_argument("--scope-r", default=1.0, type=float, help="Right scope for interpolation")
parser.add_argument("--data-choose", default=1, type=int, help="Data choice for region count")
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

args.gpus = 1 if torch.cuda.is_available() else 0
args.num_workers = 4 if args.gpus else 8

args.benchmark = True if not args.off_benchmark else False
args.is_cls_token = True if not args.off_cls_token else False
if not args.gpus:
    args.precision = 32

if args.mlp_hidden != args.hidden * 4:
    print(f"[INFO] In original paper, mlp_hidden(CURRENT:{args.mlp_hidden}) is set to: {args.hidden * 4}(={args.hidden}*4)")

train_ds, test_ds = get_dataset(args)
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
test_dl = torch.utils.data.DataLoader(test_ds, batch_size=args.eval_batch_size, num_workers=args.num_workers, pin_memory=True)


class Net(pl.LightningModule):
    def __init__(self, hparams):
        super(Net, self).__init__()
        self.save_hyperparameters(hparams)
        self.model = get_model(hparams)
        self.criterion = get_criterion(hparams)
        self.training_outputs = []  # 初始化 training_outputs
        if hparams.cutmix:
            self.cutmix = CutMix(hparams.size, beta=1.)
        if hparams.mixup:
            self.mixup = MixUp(alpha=1.)
        self.log_image_flag = hparams.api_key is None

        # 使用 torchmetrics 计算训练和验证准确率
        self.train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=hparams.num_classes)
        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=hparams.num_classes)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr, betas=(self.hparams.beta1, self.hparams.beta2), weight_decay=self.hparams.weight_decay)
        base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.max_epochs, eta_min=self.hparams.min_lr)
        scheduler = warmup_scheduler.GradualWarmupScheduler(optimizer, multiplier=1., total_epoch=self.hparams.warmup_epoch, after_scheduler=base_scheduler)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        img, label = batch
        label = label.to(torch.int64)  # 确保标签为整数类型
        if self.hparams.cutmix or self.hparams.mixup:
            if self.hparams.cutmix:
                img, label, rand_label, lambda_ = self.cutmix((img, label))
            elif self.hparams.mixup:
                if np.random.rand() <= 0.8:
                    img, label, rand_label, lambda_ = self.mixup((img, label))
                else:
                    img, label, rand_label, lambda_ = img, label, torch.zeros_like(label), 1.
            out = self.model(img)
            loss = self.criterion(out, label) * lambda_ + self.criterion(out, rand_label) * (1. - lambda_)
        else:
            out = self(img)
            loss = self.criterion(out, label)

        if not self.log_image_flag and not self.hparams.dry_run:
            self.log_image_flag = True
            self._log_image(img.clone().detach().cpu())

        self.train_accuracy.update(out, label)  # 更新训练准确率
        return {"loss": loss}

    def on_train_epoch_end(self):
        # 记录训练准确率
        train_acc = self.train_accuracy.compute()
        self.log("train/accuracy", train_acc)
        print(f"[INFO] Train Accuracy: {train_acc}")
        self.train_accuracy.reset()  # 重置准确率计算器

    def validation_step(self, batch, batch_idx):
        img, label = batch
        label = label.to(torch.int64)  # 确保标签为整数类型
        out = self(img)
        loss = self.criterion(out, label)
        self.val_accuracy.update(out, label)  # 更新验证准确率
        return {"val_loss": loss}

    def on_validation_epoch_end(self):
        # 记录验证准确率
        val_acc = self.val_accuracy.compute()
        self.log("val/accuracy", val_acc)
        print(f"[INFO] Validation Accuracy: {val_acc}")
        self.val_accuracy.reset()  # 重置准确率计算器

    def _log_image(self, image):
        grid = torchvision.utils.make_grid(image, nrow=4)
        self.logger.experiment.log_image(grid.permute(1, 2, 0))
        print("[INFO] LOG IMAGE!!!")

    def calculate_region_count(self, trainset_no_random, device):
        """Calculate the region count for the input space using the trained model."""
        self.eval()
        samples_list = []
        regions_list = []
        entropy_list = []
        for _ in range(self.hparams.average_number):
            samples = []
            random_indices = np.random.choice(len(trainset_no_random), 2, replace=False)
            for index in random_indices:
                sample, _ = trainset_no_random[index]
                samples.append(sample)
            samples_list.append(samples)

        with torch.no_grad():
            if self.hparams.data_choose <= 2:
                x_min, x_max = self.hparams.scope_l, self.hparams.scope_r
                xx = np.linspace(x_min, x_max, num=200)
                num_points = len(xx)
                for sample_1, sample_2 in samples_list:
                    generated_samples = np.zeros((num_points, 3, 32, 32))
                    for i in range(num_points):
                        alpha = xx[i]
                        generated_sample = (1 - alpha) * sample_1 + alpha * sample_2
                        generated_samples[i] = generated_sample
                    input_data = torch.tensor(generated_samples, dtype=torch.float32).to(device)
                    output = self.model(input_data)
                    _, predictions = torch.max(output, 1)
                    predictions = predictions.cpu().numpy()
                    predictions = predictions.reshape(xx.shape)
                    regions, entropy = self.cal_line(predictions)
                    regions_list.append(regions)
                    entropy_list.append(entropy)

        avg_regions = np.mean(regions_list)
        avg_entropy = np.mean(entropy_list)
        print(f"[INFO] Average regions: {avg_regions}, Average entropy: {avg_entropy}")
        return avg_regions, avg_entropy  # 返回平均 region count 和熵

    @staticmethod
    def cal_line(prediction_line):
        mark_line = np.zeros(prediction_line.shape, dtype='int64')
        mark_num = 0
        w = prediction_line.shape[0]
        direct_delta = [-1, 1]
        all_kinds = 0
        space = w
        entropy = 0
        for i in range(w):
            if mark_line[i] > 0:
                continue
            queue = [i]
            mark_num += 1
            mark_line[i] = mark_num
            tnt = 0
            while len(queue) > 0:
                tnt += 1
                cur_x = queue[0]
                queue.pop(0)
                for delta_x in direct_delta:
                    tmp_x = cur_x + delta_x
                    if tmp_x < 0 or tmp_x >= w or mark_line[tmp_x] > 0:
                        continue
                    if prediction_line[tmp_x] == prediction_line[cur_x]:
                        mark_line[tmp_x] = mark_line[cur_x]
                        queue.append(tmp_x)
            all_kinds += 1
            entropy += tnt / space * np.log(tnt / space)
        return all_kinds, -entropy


if __name__ == "__main__":
    experiment_name = get_experiment_name(args)
    print(experiment_name)
    if args.api_key:
        print("[INFO] Log with Comet.ml!")
        logger = pl.loggers.CometLogger(
            api_key=args.api_key,
            save_dir="logs",
            project_name=args.project_name,
            experiment_name=experiment_name
        )
        refresh_rate = 0
    else:
        print("[INFO] Log with CSV")
        logger = pl.loggers.CSVLogger(
            save_dir="logs",
            name=experiment_name
        )
        refresh_rate = 1
    net = Net(args)
    trainer = pl.Trainer(
        precision=args.precision,
        fast_dev_run=args.dry_run,
        devices=[int(found_device())],  # 强制使用单 GPU
        accelerator='gpu' if args.gpus > 0 else 'cpu',
        benchmark=args.benchmark,
        logger=logger,
        max_epochs=args.max_epochs,
        enable_progress_bar=True
    )
    trainer.fit(model=net, train_dataloaders=train_dl, val_dataloaders=test_dl)
    if not args.dry_run:
        model_path = f"weights/{experiment_name}.pth"
        torch.save(net.state_dict(), model_path)
        if args.api_key:
            logger.experiment.log_asset(file_name=experiment_name, file_data=model_path)

    # After training, calculate the region count
    net.to(device)  # 将模型移动到正确的设备
    avg_regions, avg_entropy = net.calculate_region_count(train_ds, device)

    # Open the file in append mode
    f = open('region_count.txt', 'a')

    # Format the training and validation accuracy to 4 decimal places and write them to the file
    f.write(
        f"{net.trainer.logged_metrics['train/accuracy']:.4f} "
        f"{net.trainer.logged_metrics['val/accuracy']:.4f} "
        f"{avg_regions} {args.lr} {args.batch_size} {args.weight_decay}\n"
    )

    # Close the file
    f.close()
