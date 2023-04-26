# Fine-tuning MT5 model using Deepspeed 
## with Accelerator
* Multi-GPU
```
accelerate launch --config_file deepspeed_config_zero2.yaml accelerate_deepspeed.py
```

## through Transformers source codes
* TODO Multi-GPU training seems not to be supported.
```
deepspeed transformers_deepspeed.py
```
