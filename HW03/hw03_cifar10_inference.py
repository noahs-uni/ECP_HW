import os
import zipfile
import torch

import requests
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import CIFAR10
from tqdm import tqdm
import torch.nn as nn

from models.resnet import resnet18, resnet34, resnet50
from models.vgg import vgg11_bn, vgg13_bn, vgg19_bn
from models.densenet import densenet121, densenet161, densenet169
from models.inception import inception_v3 # slow, propably bad cifar10 implementation of inception for PT

from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib

import timeit

threads = 40
torch.set_num_threads(threads)

#maybe better performance
# %env OMP_PLACES=cores
os.environ['OMP_PLACES'] = 'cores'
# %env OMP_PROC_BIND=close
os.environ['OMP_PROC_BIND'] = 'close'
# %env OMP_WAIT_POLICY=active
os.environ['OMP_WAIT_POLICY'] = 'active'

threads = 40
torch.set_num_threads(threads)

# dataloader
def val_dataloader(mean = (0.4914, 0.4822, 0.4465), std = (0.2471, 0.2435, 0.2616)):

    transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )
    dataset = CIFAR10(root="datasets/cifar10_data", train=False, download=True, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=128,
        num_workers=0,
        drop_last=True,
        pin_memory=False,
    )
    return dataloader


# model calibration for quantization
def collect_stats(model, data_loader, num_batches):
     """Feed data to the network and collect statistic"""

     # Enable calibrators
     for name, module in model.named_modules():
         if isinstance(module, quant_nn.TensorQuantizer):
             if module._calibrator is not None:
                 module.disable_quant()
                 module.enable_calib()
             else:
                 module.disable()

     for i, (image, _) in tqdm(enumerate(data_loader), total=num_batches):
         model(image.cpu())
         if i >= num_batches:
             break

     # Disable calibrators
     for name, module in model.named_modules():
         if isinstance(module, quant_nn.TensorQuantizer):
             if module._calibrator is not None:
                 module.enable_quant()
                 module.disable_calib()
             else:
                 module.enable()


def compute_amax(model, **kwargs):
 # Load calib result
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax()
                else:
                    module.load_calib_amax(**kwargs)
            print(F"{name:40}: {module}")
    model.cpu()

# log
def log_to_file(time_log, acc, multiplier_name, model_name, filepath):
    try:
        with open(filepath, 'a') as f:
            f.write(f"multiplier: {multiplier_name}\n")
            f.write(f"model: {model_name}\n")
            f.write('Inference time: %.4f s\n' % time_log)
            f.write('Accuracy of the network on the 10000 test images: %.4f %%\n' % acc)
            f.write("=================================================\n")
            
        print(f"\nLogged to: {filepath}")
        
    except Exception as e:
        print(f"Failed to write to log file: {e}")

def main():
    # hw for group 05
    axx_mult_list = ['SPR_9_62', 'SPR_12_52']
    for axx_mult in axx_mult_list:
        model_1 = resnet34(pretrained=True, axx_mult = axx_mult)
        model_2 = densenet121(pretrained=True, axx_mult = axx_mult)
        model_list = [model_1, model_2]
        model_name_list = ['resnet34', 'densenet121']
        for model, model_name in zip(model_list, model_name_list):
            model.eval()

            # load dataset
            transform = T.Compose(
                    [
                        T.RandomCrop(32, padding=4),
                        T.RandomHorizontalFlip(),
                        T.ToTensor(),
                        T.Normalize(mean = (0.4914, 0.4822, 0.4465), std = (0.2471, 0.2435, 0.2616)),
                    ]
                )
            dataset = CIFAR10(root="datasets/cifar10_data", train=True, download=True, transform=transform)

            evens = list(range(0, len(dataset), 10))
            trainset_1 = torch.utils.data.Subset(dataset, evens)

            data = val_dataloader()
            # data_t is used for calibration purposes and is a subset of train-set
            data_t = DataLoader(trainset_1, batch_size=128,
                                                        shuffle=False, num_workers=0)
            
            # Calibrates the quantization parameters
            # It is a bit slow since we collect histograms on CPU
            with torch.no_grad():
                stats = collect_stats(model, data_t, num_batches=2)
                amax = compute_amax(model, method="percentile", percentile=99.99)
                
                # optional - test different calibration methods
                #amax = compute_amax(model, method="mse")
                #amax = compute_amax(model, method="entropy")
            
            # model evaluation    
            correct = 0
            total = 0

            model.eval()
            start_time = timeit.default_timer()
            with torch.no_grad():
                for iteraction, (images, labels) in tqdm(enumerate(data), total=len(data)):
                    images, labels = images.to("cpu"), labels.to("cpu")
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            time_log = timeit.default_timer() - start_time
            acc = 100 * correct / total
            print(time_log)
            print('Accuracy of the network on the 10000 test images: %.4f %%' % acc)
            log_to_file(
                time_log=time_log,
                acc=acc,
                multiplier_name=axx_mult,
                model_name=model_name,
                filepath='model_inference_result.log'
            )
        

if __name__ == "__main__":
    main()