To download the required checkpoints, please go to [THIS LINK](https://drive.google.com/drive/folders/1WFAHaEXwefBslwtSQWUVEwFi8FTtuu5c?usp=sharing).

Then put all the downloaded files under the `ckpt` directory, except the checkpoint under folder `floortrans/models`, which should be manually moved to the designated location. You should have a directory tree like this:

```
└── ckpt
    ├── CubiCasa5k
    │   └── model_best_val_loss_var.pkl
    ├── CVCFP_stairs
    │   ├── 1
    │   │   └── model_final.pth
    │   ├── 2
    │   │   └── model_final.pth
    │   ├── 3
    │   │   └── model_final.pth
    │   ├── 4
    │   │   └── model_final.pth
    │   └── 5
    │       └── model_final.pth
    ├── CVCFP_wdw
    │   ├── 1
    │   │   └── model_0009999.pth
    │   ├── 2
    │   │   └── model_0009999.pth
    │   ├── 3
    │   │   └── model_0009999.pth
    │   ├── 4
    │   │   └── model_0009999.pth
    │   └── 5
    │       └── model_0009999.pth
    └── text
        ├── text_seg_loss_590epos.pt
        ├── text_seg_loss_681epos.pt
        ├── text_seg_lr_590epos.pt
        ├── text_seg_lr_681epos.pt
        ├── text_seg_model_590epos.pt
        ├── text_seg_model_681epos.pt
        ├── text_seg_optim_590epos.pt
        └── text_seg_optim_681epos.pt
...
└── floortrans
    └── models
        └── model_1427.pth
        ...
```
