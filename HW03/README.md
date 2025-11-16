**CIFAR-10 Inference** with `hw03_cifar10_inference.py`

+ Place script `hw03_cifar10_inference.py` into `adapt/examples/` and run **inside the Docker or Apptainer image**
+ Place header files into `adapt/adapt/cpu-kernels/axx_mults`

**CIFAR-10 Re-training** with `hw03_cifar10_retraining.py`

+ Place script `hw03_cifar10_retraining.py` into `adapt/examples/` and run **inside the Docker or Apptainer image**
+ Model weights are saved to `adapt/examples/models/state_dicts`

Attach to Apptainer in cluster
```
tmux attach-session -t hw03_ecp
```
