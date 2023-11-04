
# Pool Boiling Gravity


## G-FNO

```
CUDA_VISIBLE_DEVICES=0 python train.py data_base_dir=/nvme-data/jacob/BubbleML log_dir=/mnt/data/shared/jacob/GFNO/BubbleML/logs/PB_Gravity/gfno/full_modes_pad_upscale dataset=PB_Gr
avity experiment=gfno/pb_temp_full
data_base_dir: /nvme-data/jacob/BubbleML
log_dir: /mnt/data/shared/jacob/GFNO/BubbleML/logs/PB_Gravity/gfno/full_modes_pad_upscale
train: true
test: true
model_checkpoint: null
dataset:
  name: pb_gravity
  transform: true
  steady_time: 30
  train_paths:
  - ${data_base_dir}/PoolBoiling-Gravity-FC72-2D/gravY-0.0001.hdf5
  - ${data_base_dir}/PoolBoiling-Gravity-FC72-2D/gravY-0.001.hdf5
  - ${data_base_dir}/PoolBoiling-Gravity-FC72-2D/gravY-0.02.hdf5
  - ${data_base_dir}/PoolBoiling-Gravity-FC72-2D/gravY-0.05.hdf5
  - ${data_base_dir}/PoolBoiling-Gravity-FC72-2D/gravY-0.1.hdf5
  - ${data_base_dir}/PoolBoiling-Gravity-FC72-2D/gravY-0.5.hdf5
  - ${data_base_dir}/PoolBoiling-Gravity-FC72-2D/gravY-1.0.hdf5
  val_paths:
  - ${data_base_dir}/PoolBoiling-Gravity-FC72-2D/gravY-0.2.hdf5
experiment:
  torch_dataset_name: temp_input_dataset
  distributed: false
  train:
    max_epochs: 250
    batch_size: 8
    shuffle_data: true
    time_window: 5
    future_window: 5
    push_forward_steps: 1
    use_coords: true
    noise: true
    downsample_factor: 1
  model:
    model_name: gfno
    fmode_frac:
    - 0.165
    - 0.165
    width: 32
    reflection: false
    domain_padding: 0.1
  optimizer:
    initial_lr: 0.001
    weight_decay: 1.0e-06
  lr_scheduler:
    name: step
    patience: 75
    factor: 0.5

['${data_base_dir}/PoolBoiling-Gravity-FC72-2D/gravY-0.0001.hdf5', '${data_base_dir}/PoolBoiling-Gravity-FC72-2D/gravY-0.001.hdf5', '${data_base_dir}/PoolBoiling-Gravity-FC72-2D/gravY-0.02.hdf5', '${data_base_dir}/PoolBoiling-Gravity-FC72-2D/gravY-0.05.hdf5', '
${data_base_dir}/PoolBoiling-Gravity-FC72-2D/gravY-0.1.hdf5', '${data_base_dir}/PoolBoiling-Gravity-FC72-2D/gravY-0.5.hdf5', '${data_base_dir}/PoolBoiling-Gravity-FC72-2D/gravY-1.0.hdf5']
train size:  142
# parameters: 112936965
GFNO2d(
  (domain_padding): DomainPadding()
  (p): GConv2d()
  (conv0): GSpectralConv2d(
    (conv): GConv2d(
      (W): ParameterDict(
          (00_modes): Parameter containing: [torch.cuda.FloatTensor of size 32x1x32x4x1x1 (GPU 0)]
          (y0_modes): Parameter containing: [torch.cuda.ComplexFloatTensor of size 32x1x32x4x41x1 (GPU 0)]
          (yposx_modes): Parameter containing: [torch.cuda.ComplexFloatTensor of size 32x1x32x4x83x41 (GPU 0)]
      )
    )
  )
  (conv1): GSpectralConv2d(
    (conv): GConv2d(
      (W): ParameterDict(
          (00_modes): Parameter containing: [torch.cuda.FloatTensor of size 32x1x32x4x1x1 (GPU 0)]
          (y0_modes): Parameter containing: [torch.cuda.ComplexFloatTensor of size 32x1x32x4x41x1 (GPU 0)]
          (yposx_modes): Parameter containing: [torch.cuda.ComplexFloatTensor of size 32x1x32x4x83x41 (GPU 0)]
      )
    )
  )
  (conv2): GSpectralConv2d(
    (conv): GConv2d(
      (W): ParameterDict(
          (00_modes): Parameter containing: [torch.cuda.FloatTensor of size 32x1x32x4x1x1 (GPU 0)]
          (y0_modes): Parameter containing: [torch.cuda.ComplexFloatTensor of size 32x1x32x4x41x1 (GPU 0)]
          (yposx_modes): Parameter containing: [torch.cuda.ComplexFloatTensor of size 32x1x32x4x83x41 (GPU 0)]
      )
    )
  )
  (conv3): GSpectralConv2d(
    (conv): GConv2d(
      (W): ParameterDict(
          (00_modes): Parameter containing: [torch.cuda.FloatTensor of size 32x1x32x4x1x1 (GPU 0)]
          (y0_modes): Parameter containing: [torch.cuda.ComplexFloatTensor of size 32x1x32x4x41x1 (GPU 0)]
          (yposx_modes): Parameter containing: [torch.cuda.ComplexFloatTensor of size 32x1x32x4x83x41 (GPU 0)]
      )
    )
  )
  (mlp0): GMLP2d(
    (mlp1): GConv2d()
    (mlp2): GConv2d()
  )
  (mlp1): GMLP2d(
    (mlp1): GConv2d()
    (mlp2): GConv2d()
  )
  (mlp2): GMLP2d(
    (mlp1): GConv2d()
    (mlp2): GConv2d()
  )
  (mlp3): GMLP2d(
    (mlp1): GConv2d()
    (mlp2): GConv2d()
  )
  (w0): GConv2d()
  (w1): GConv2d()
  (w2): GConv2d()
  (w3): GConv2d()
  (norm): GNorm(
    (norm): InstanceNorm3d(32, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
  )
  (q): GMLP2d(
    (mlp1): GConv2d()
    (mlp2): GConv2d()
  )
)
Model has 56510469 parameters
Model has 112936965 parameters
<op_lib.temp_trainer.TempTrainer object at 0x7fcbe8bb0b20>
epoch  0
...
saving model to /mnt/data/shared/jacob/GFNO/BubbleML/logs/PB_Gravity/gfno/full_modes_pad_upscale/pb_gravity/GFNO2d_pb_gravity_temp_input_dataset_1699005642.pt
rollout time 3.7529025077819824 (s)
tensor(1.) tensor(-1.)
tensor(1.0012) tensor(-1.0004)
512 512
torch.Size([165, 2048])

            MAE: 0.04727041348814964
            RMSE: 0.11084376275539398
            Relative Error: 0.11668718606233597
            Max Error: 3.6429691314697266
            Boundary RMSE: 0.20443813502788544
            Interface RMSE: 0.4692804515361786
            Fourier
                - Low: 1.2311668395996094
                - Mid: 0.7548862099647522
                - High: 0.06600255519151688

rollout time 3.770886182785034 (s)
tensor(1.) tensor(-1.)
tensor(1.) tensor(-1.)
512 512
torch.Size([165, 2048])

            MAE: 0.0
            RMSE: 0.0
            Relative Error: 0.0
            Max Error: 0.0
            Boundary RMSE: 0.0
            Interface RMSE: 0.0
            Fourier
                - Low: 0.0
                - Mid: 0.0
                - High: 0.0


            MAE: 0.0
            RMSE: 0.0
            Relative Error: 0.0
            Max Error: 0.0
            Boundary RMSE: 0.0
            Interface RMSE: 0.0
            Fourier
                - Low: 0.0
                - Mid: 0.0
                - High: 0.0
```

## FNO

```
CUDA_VISIBLE_DEVICES=2 python train.py data_base_dir=/data/jacob/BubbleML log_dir=/mnt/data/shared/jacob/GFNO/BubbleML/logs/PB_Gravity/fno/ dataset=PB_Gravity experiment=fno/pb_temp
_full
data_base_dir: /data/jacob/BubbleML
log_dir: /mnt/data/shared/jacob/GFNO/BubbleML/logs/PB_Gravity/fno/
train: true
test: true
model_checkpoint: null
dataset:
  name: pb_gravity
  transform: true
  steady_time: 30
  train_paths:
  - ${data_base_dir}/PoolBoiling-Gravity-FC72-2D/gravY-0.0001.hdf5
  - ${data_base_dir}/PoolBoiling-Gravity-FC72-2D/gravY-0.001.hdf5
  - ${data_base_dir}/PoolBoiling-Gravity-FC72-2D/gravY-0.02.hdf5
  - ${data_base_dir}/PoolBoiling-Gravity-FC72-2D/gravY-0.05.hdf5
  - ${data_base_dir}/PoolBoiling-Gravity-FC72-2D/gravY-0.1.hdf5
  - ${data_base_dir}/PoolBoiling-Gravity-FC72-2D/gravY-0.5.hdf5
  - ${data_base_dir}/PoolBoiling-Gravity-FC72-2D/gravY-1.0.hdf5
  val_paths:
  - ${data_base_dir}/PoolBoiling-Gravity-FC72-2D/gravY-0.2.hdf5
experiment:
  torch_dataset_name: temp_input_dataset
  distributed: false
  train:
    max_epochs: 250
    batch_size: 8
    shuffle_data: true
    time_window: 5
    future_window: 5
    push_forward_steps: 1
    use_coords: true
    noise: true
    downsample_factor: 1
  model:
    model_name: fno
    fmode_frac:
    - 0.165
    - 0.165
    hidden_channels: 64
    domain_padding:
    - 0.1
    - 0.1
    n_layers: 4
    norm: group_norm
  optimizer:
    initial_lr: 0.001
    weight_decay: 1.0e-06
  lr_scheduler:
    name: step
    patience: 75
    factor: 0.5

['${data_base_dir}/PoolBoiling-Gravity-FC72-2D/gravY-0.0001.hdf5', '${data_base_dir}/PoolBoiling-Gravity-FC72-2D/gravY-0.001.hdf5', '${data_base_dir}/PoolBoiling-Gravity-FC72-2D/gravY-0.02.hdf5', '${data_base_dir}/PoolBoiling-Gravity-FC72-2D/gravY-0.05.hdf5', '${data_base_dir}/PoolBoiling-Gravity-FC72-2D/gravY-0.1.hdf5', '${data_base_dir}/PoolBoiling-Gravity-FC72-2D/gravY-0.5.hdf5', '${data_base_dir}/PoolBoiling-Gravity-FC72-2D/gravY-1.0.hdf5']
train size:  142
FNO(
  (domain_padding): DomainPadding()
  (fno_blocks): FNOBlocks(
    (convs): FactorizedSpectralConv(
      (weight): ModuleList(
        (0): ComplexDenseTensor(shape=torch.Size([64, 64, 42, 42]), rank=None)
        (1): ComplexDenseTensor(shape=torch.Size([64, 64, 42, 42]), rank=None)
        (2): ComplexDenseTensor(shape=torch.Size([64, 64, 42, 42]), rank=None)
        (3): ComplexDenseTensor(shape=torch.Size([64, 64, 42, 42]), rank=None)
        (4): ComplexDenseTensor(shape=torch.Size([64, 64, 42, 42]), rank=None)
        (5): ComplexDenseTensor(shape=torch.Size([64, 64, 42, 42]), rank=None)
        (6): ComplexDenseTensor(shape=torch.Size([64, 64, 42, 42]), rank=None)
        (7): ComplexDenseTensor(shape=torch.Size([64, 64, 42, 42]), rank=None)
      )
    )
    (fno_skips): ModuleList(
      (0): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (2): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (3): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
    )
    (norm): ModuleList(
      (0): GroupNorm(1, 64, eps=1e-05, affine=True)
      (1): GroupNorm(1, 64, eps=1e-05, affine=True)
      (2): GroupNorm(1, 64, eps=1e-05, affine=True)
      (3): GroupNorm(1, 64, eps=1e-05, affine=True)
    )
  )
  (lifting): Lifting(
    (fc): Conv2d(27, 64, kernel_size=(1, 1), stride=(1, 1))
  )
  (projection): Projection(
    (fc1): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1))
    (fc2): Conv2d(256, 5, kernel_size=(1, 1), stride=(1, 1))
  )
)
Model has 115642373 parameters
Model has 115642373 parameters
<op_lib.temp_trainer.TempTrainer object at 0x7fa16a74b1c0>
epoch  0
...
saving model to /mnt/data/shared/jacob/GFNO/BubbleML/logs/PB_Gravity/fno/pb_gravity/FNO_pb_gravity_temp_input_dataset_1698993425.pt
rollout time 9.345260381698608 (s)
tensor(1.) tensor(-1.)
tensor(1.0012) tensor(-1.0004)
512 512
torch.Size([165, 2048])

            MAE: 0.04719358682632446
            RMSE: 0.10667844861745834
            Relative Error: 0.11233281344175339
            Max Error: 3.8274190425872803
            Boundary RMSE: 0.1980457454919815
            Interface RMSE: 0.44389089941978455
            Fourier
                - Low: 1.137291431427002
                - Mid: 0.746132493019104
                - High: 0.06388316303491592

rollout time 9.030315399169922 (s)
tensor(1.) tensor(-1.)
tensor(1.) tensor(-1.)
512 512
torch.Size([165, 2048])

            MAE: 0.04719358682632446
            RMSE: 0.10667844861745834
            Relative Error: 0.11233281344175339
            Max Error: 3.8274190425872803
            Boundary RMSE: 0.1980457454919815
            Interface RMSE: 0.44389089941978455
            Fourier
                - Low: 1.137291431427002
                - Mid: 0.746132493019104
                - High: 0.06388316303491592

rollout time 9.030315399169922 (s)
tensor(1.) tensor(-1.)
tensor(1.) tensor(-1.)
512 512
torch.Size([165, 2048])

            MAE: 0.0
            RMSE: 0.0
            Relative Error: 0.0
            Max Error: 0.0
            Boundary RMSE: 0.0
            Interface RMSE: 0.0
            Fourier
                - Low: 0.0
                - Mid: 0.0
                - High: 0.0


            MAE: 0.0
            RMSE: 0.0
            Relative Error: 0.0
            Max Error: 0.0
            Boundary RMSE: 0.0
            Interface RMSE: 0.0
            Fourier
                - Low: 0.0
                - Mid: 0.0
                - High: 0.0
```

# Pool Boiling Saturated

## G-FNO
```
CUDA_VISIBLE_DEVICES=4 python train.py data_base_dir=/nvme-data/jacob/BubbleML log_dir=/mnt/data/shared/jacob/GFNO/BubbleML/logs/PB_Saturated/gfno/full_modes_pad_upscale dataset=PB_WallSuperHeat experiment=gfno/pb_temp_full
data_base_dir: /nvme-data/jacob/BubbleML
log_dir: /mnt/data/shared/jacob/GFNO/BubbleML/logs/PB_Saturated/gfno/full_modes_pad_upscale
train: true
test: true
model_checkpoint: null
dataset:
  name: wall_super_heat
  transform: true
  steady_time: 30
  train_paths:
  - ${data_base_dir}/PoolBoiling-WallSuperheat-FC72-2D/Twall-60.hdf5
  - ${data_base_dir}/PoolBoiling-WallSuperheat-FC72-2D/Twall-65.hdf5
  - ${data_base_dir}/PoolBoiling-WallSuperheat-FC72-2D/Twall-70.hdf5
  - ${data_base_dir}/PoolBoiling-WallSuperheat-FC72-2D/Twall-75.hdf5
  - ${data_base_dir}/PoolBoiling-WallSuperheat-FC72-2D/Twall-80.hdf5
  - ${data_base_dir}/PoolBoiling-WallSuperheat-FC72-2D/Twall-85.hdf5
  - ${data_base_dir}/PoolBoiling-WallSuperheat-FC72-2D/Twall-90.hdf5
  - ${data_base_dir}/PoolBoiling-WallSuperheat-FC72-2D/Twall-100.hdf5
  - ${data_base_dir}/PoolBoiling-WallSuperheat-FC72-2D/Twall-105.hdf5
  - ${data_base_dir}/PoolBoiling-WallSuperheat-FC72-2D/Twall-110.hdf5
  - ${data_base_dir}/PoolBoiling-WallSuperheat-FC72-2D/Twall-115.hdf5
  - ${data_base_dir}/PoolBoiling-WallSuperheat-FC72-2D/Twall-120.hdf5
  val_paths:
  - ${data_base_dir}/PoolBoiling-WallSuperheat-FC72-2D/Twall-95.hdf5
experiment:
  torch_dataset_name: temp_input_dataset
  distributed: false
  train:
    max_epochs: 250
    batch_size: 8
    shuffle_data: true
    time_window: 5
    future_window: 5
    push_forward_steps: 1
    use_coords: true
    noise: true
    downsample_factor: 1
  model:
    model_name: gfno
    fmode_frac:
    - 0.165
    - 0.165
    width: 32
    reflection: false
    domain_padding: 0.1
  optimizer:
    initial_lr: 0.001
    weight_decay: 1.0e-06
  lr_scheduler:
    name: step
    patience: 75
    factor: 0.5

['${data_base_dir}/PoolBoiling-WallSuperheat-FC72-2D/Twall-60.hdf5', '${data_base_dir}/PoolBoiling-WallSuperheat-FC72-2D/Twall-65.hdf5', '${data_base_dir}/PoolBoiling-WallSuperheat-FC72-2D/Twall-70.hdf5', '${data_base_dir}/PoolBoiling-WallSuperheat-FC72-2D/Twall-75.hdf5', '${data_base_dir}/PoolBoiling-WallSuperheat-FC72-2D/Twall-80.hdf5', '${data_base_dir}/PoolBoiling-WallSuperheat-FC72-2D/Twall-85.hdf5', '${data_base_dir}/PoolBoiling-WallSuperheat-FC72-2D/Twall-90.hdf5', '${data_base_dir}/PoolBoiling-WallSuperheat-FC72-2D/Twall-100.hdf5', '${data_base_dir}/PoolBoiling-WallSuperheat-FC72-2D/Twall-105.hdf5', '${data_base_dir}/PoolBoiling-WallSuperheat-FC72-2D/Twall-110.hdf5', '${data_base_dir}/PoolBoiling-WallSuperheat-FC72-2D/Twall-115.hdf5', '${data_base_dir}/PoolBoiling-WallSuperheat-FC72-2D/Twall-120.hdf5']
wall temp tensor(93.9168, dtype=torch.float64)
train size:  243
# parameters: 112936965
GFNO2d(
  (domain_padding): DomainPadding()
  (p): GConv2d()
  (conv0): GSpectralConv2d(
    (conv): GConv2d(
      (W): ParameterDict(
          (00_modes): Parameter containing: [torch.cuda.FloatTensor of size 32x1x32x4x1x1 (GPU 0)]
          (y0_modes): Parameter containing: [torch.cuda.ComplexFloatTensor of size 32x1x32x4x41x1 (GPU 0)]
          (yposx_modes): Parameter containing: [torch.cuda.ComplexFloatTensor of size 32x1x32x4x83x41 (GPU 0)]
      )
    )
  )
  (conv1): GSpectralConv2d(
    (conv): GConv2d(
      (W): ParameterDict(
          (00_modes): Parameter containing: [torch.cuda.FloatTensor of size 32x1x32x4x1x1 (GPU 0)]
          (y0_modes): Parameter containing: [torch.cuda.ComplexFloatTensor of size 32x1x32x4x41x1 (GPU 0)]
          (yposx_modes): Parameter containing: [torch.cuda.ComplexFloatTensor of size 32x1x32x4x83x41 (GPU 0)]
      )
    )
  )
  (conv2): GSpectralConv2d(
    (conv): GConv2d(
      (W): ParameterDict(
          (00_modes): Parameter containing: [torch.cuda.FloatTensor of size 32x1x32x4x1x1 (GPU 0)]
          (y0_modes): Parameter containing: [torch.cuda.ComplexFloatTensor of size 32x1x32x4x41x1 (GPU 0)]
          (yposx_modes): Parameter containing: [torch.cuda.ComplexFloatTensor of size 32x1x32x4x83x41 (GPU 0)]
      )
    )
  )
  (conv3): GSpectralConv2d(
    (conv): GConv2d(
      (W): ParameterDict(
          (00_modes): Parameter containing: [torch.cuda.FloatTensor of size 32x1x32x4x1x1 (GPU 0)]
          (y0_modes): Parameter containing: [torch.cuda.ComplexFloatTensor of size 32x1x32x4x41x1 (GPU 0)]
          (yposx_modes): Parameter containing: [torch.cuda.ComplexFloatTensor of size 32x1x32x4x83x41 (GPU 0)]
      )
    )
  )
  (mlp0): GMLP2d(
    (mlp1): GConv2d()
    (mlp2): GConv2d()
  )
  (mlp1): GMLP2d(
    (mlp1): GConv2d()
    (mlp2): GConv2d()
  )
  (mlp2): GMLP2d(
    (mlp1): GConv2d()
    (mlp2): GConv2d()
  )
  (mlp3): GMLP2d(
    (mlp1): GConv2d()
    (mlp2): GConv2d()
  )
  (w0): GConv2d()
  (w1): GConv2d()
  (w2): GConv2d()
  (w3): GConv2d()
  (norm): GNorm(
    (norm): InstanceNorm3d(32, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
  )
  (q): GMLP2d(
    (mlp1): GConv2d()
    (mlp2): GConv2d()
  )
)
Model has 56510469 parameters
Model has 112936965 parameters
<op_lib.temp_trainer.TempTrainer object at 0x7f81e3d91880>
epoch  0
...
saving model to /mnt/data/shared/jacob/GFNO/BubbleML/logs/PB_Saturated/gfno/full_modes_pad_upscale/wall_super_heat/GFNO2d_wall_super_heat_temp_input_dataset_1699063673.pt
rollout time 4.732235908508301 (s)
tensor(0.8341) tensor(-1.)
tensor(0.5757) tensor(-1.0180)
512 512
torch.Size([165, 2048])

            MAE: 0.01649664342403412
            RMSE: 0.07010267674922943
            Relative Error: 0.07116028666496277
            Max Error: 3.0584604740142822
            Boundary RMSE: 0.21343809366226196
            Interface RMSE: 0.20237061381340027
            Fourier
                - Low: 0.44889017939567566
                - Mid: 0.5475609302520752
                - High: 0.06319791823625565
        
rollout time 3.3405258655548096 (s)
tensor(0.8341) tensor(-1.)
tensor(0.8341) tensor(-1.)
512 512
torch.Size([165, 2048])

            MAE: 0.0
            RMSE: 0.0
            Relative Error: 0.0
            Max Error: 0.0
            Boundary RMSE: 0.0
            Interface RMSE: 0.0
            Fourier
                - Low: 0.0
                - Mid: 0.0
                - High: 0.0
        

            MAE: 0.0
            RMSE: 0.0
            Relative Error: 0.0
            Max Error: 0.0
            Boundary RMSE: 0.0
            Interface RMSE: 0.0
            Fourier
                - Low: 0.0
                - Mid: 0.0
                - High: 0.0
```

## FNO

```
 CUDA_VISIBLE_DEVICES=2 python train.py data_base_dir=/data/jacob/BubbleML log_dir=/mnt/data/shared/jacob/GFNO/BubbleML/logs/PB_Saturated/fno/ dataset=PB_WallSuperHeat experiment=fno/pb_temp_full
data_base_dir: /data/jacob/BubbleML
log_dir: /mnt/data/shared/jacob/GFNO/BubbleML/logs/PB_Saturated/fno/
train: true
test: true
model_checkpoint: null
dataset:
  name: wall_super_heat
  transform: true
  steady_time: 30
  train_paths:
  - ${data_base_dir}/PoolBoiling-WallSuperheat-FC72-2D/Twall-60.hdf5
  - ${data_base_dir}/PoolBoiling-WallSuperheat-FC72-2D/Twall-65.hdf5
  - ${data_base_dir}/PoolBoiling-WallSuperheat-FC72-2D/Twall-70.hdf5
  - ${data_base_dir}/PoolBoiling-WallSuperheat-FC72-2D/Twall-75.hdf5
  - ${data_base_dir}/PoolBoiling-WallSuperheat-FC72-2D/Twall-80.hdf5
  - ${data_base_dir}/PoolBoiling-WallSuperheat-FC72-2D/Twall-85.hdf5
  - ${data_base_dir}/PoolBoiling-WallSuperheat-FC72-2D/Twall-90.hdf5
  - ${data_base_dir}/PoolBoiling-WallSuperheat-FC72-2D/Twall-100.hdf5
  - ${data_base_dir}/PoolBoiling-WallSuperheat-FC72-2D/Twall-105.hdf5
  - ${data_base_dir}/PoolBoiling-WallSuperheat-FC72-2D/Twall-110.hdf5
  - ${data_base_dir}/PoolBoiling-WallSuperheat-FC72-2D/Twall-115.hdf5
  - ${data_base_dir}/PoolBoiling-WallSuperheat-FC72-2D/Twall-120.hdf5
  val_paths:
  - ${data_base_dir}/PoolBoiling-WallSuperheat-FC72-2D/Twall-95.hdf5
experiment:
  torch_dataset_name: temp_input_dataset
  distributed: false
  train:
    max_epochs: 250
    batch_size: 8
    shuffle_data: true
    time_window: 5
    future_window: 5
    push_forward_steps: 1
    use_coords: true
    noise: true
    downsample_factor: 1
  model:
    model_name: fno
    fmode_frac:
    - 0.165
    - 0.165
    hidden_channels: 64
    domain_padding:
    - 0.1
    - 0.1
    n_layers: 4
    norm: group_norm
  optimizer:
    initial_lr: 0.001
    weight_decay: 1.0e-06
      lr_scheduler:
    name: step
    patience: 75
    factor: 0.5

['${data_base_dir}/PoolBoiling-WallSuperheat-FC72-2D/Twall-60.hdf5', '${data_base_dir}/PoolBoiling-WallSuperheat-FC72-2D/Twall-65.hdf5', '${data_base_dir}/PoolBoiling-WallSuperheat-FC72-2D/Twall-70.hdf5', '${data_base_dir}/PoolBoiling-WallSuperheat-FC72-2D/Twall-75.hdf5', '${data_base_dir}/PoolBoiling-WallSuperheat-FC72-2D/Twall-80.hdf5', '${data_base_dir}/PoolBoiling-WallSuperheat-FC72-2D/Twall-85.hdf5', '${data_base_dir}/PoolBoiling-WallSuperheat-FC72-2D/Twall-90.hdf5', '${data_base_dir}/PoolBoiling-WallSuperheat-FC72-2D/Twall-100.hdf5', '${data_base_dir}/PoolBoiling-WallSuperheat-FC72-2D/Twall-105.hdf5', '${data_base_dir}/PoolBoiling-WallSuperheat-FC72-2D/Twall-110.hdf5', '${data_base_dir}/PoolBoiling-WallSuperheat-FC72-2D/Twall-115.hdf5', '${data_base_dir}/PoolBoiling-WallSuperheat-FC72-2D/Twall-120.hdf5']
wall temp tensor(93.9168, dtype=torch.float64)
train size:  243
FNO(
  (domain_padding): DomainPadding()
  (fno_blocks): FNOBlocks(
    (convs): FactorizedSpectralConv(
      (weight): ModuleList(
        (0): ComplexDenseTensor(shape=torch.Size([64, 64, 42, 42]), rank=None)
        (1): ComplexDenseTensor(shape=torch.Size([64, 64, 42, 42]), rank=None)
        (2): ComplexDenseTensor(shape=torch.Size([64, 64, 42, 42]), rank=None)
        (3): ComplexDenseTensor(shape=torch.Size([64, 64, 42, 42]), rank=None)
        (4): ComplexDenseTensor(shape=torch.Size([64, 64, 42, 42]), rank=None)
        (5): ComplexDenseTensor(shape=torch.Size([64, 64, 42, 42]), rank=None)
        (6): ComplexDenseTensor(shape=torch.Size([64, 64, 42, 42]), rank=None)
        (7): ComplexDenseTensor(shape=torch.Size([64, 64, 42, 42]), rank=None)
      )
    )
    (fno_skips): ModuleList(
      (0): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (2): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (3): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
    )
    (norm): ModuleList(
      (0): GroupNorm(1, 64, eps=1e-05, affine=True)
      (1): GroupNorm(1, 64, eps=1e-05, affine=True)
      (2): GroupNorm(1, 64, eps=1e-05, affine=True)
      (3): GroupNorm(1, 64, eps=1e-05, affine=True)
    )
  )
  (lifting): Lifting(
    (fc): Conv2d(27, 64, kernel_size=(1, 1), stride=(1, 1))
  )
  (projection): Projection(
    (fc1): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1))
    (fc2): Conv2d(256, 5, kernel_size=(1, 1), stride=(1, 1))
  )
)
Model has 115642373 parameters
Model has 115642373 parameters
<op_lib.temp_trainer.TempTrainer object at 0x7f1ef9d5d220>
epoch  0
...
saving model to /mnt/data/shared/jacob/GFNO/BubbleML/logs/PB_Saturated/fno/wall_super_heat/FNO_wall_super_heat_temp_input_dataset_1699042533.pt
rollout time 4.629623174667358 (s)
tensor(0.6487) tensor(-1.)
tensor(0.5757) tensor(-1.0180)
512 512
torch.Size([165, 2048])

            MAE: 0.017788276076316833
            RMSE: 0.07655420899391174
            Relative Error: 0.07770612835884094
            Max Error: 2.279771566390991
            Boundary RMSE: 0.21660102903842926
            Interface RMSE: 0.15318486094474792
            Fourier
                - Low: 0.6040723323822021
                - Mid: 0.6084242463111877
                - High: 0.06291680037975311

rollout time 4.332693338394165 (s)
tensor(0.6487) tensor(-1.)
tensor(0.6487) tensor(-1.)
512 512
torch.Size([165, 2048])

            MAE: 0.0
            RMSE: 0.0
            Relative Error: 0.0
            Max Error: 0.0
            Boundary RMSE: 0.0
            Interface RMSE: 0.0
            Fourier
                - Low: 0.0
                - Mid: 0.0
                - High: 0.0


            MAE: 0.0
            RMSE: 0.0
            Relative Error: 0.0
            Max Error: 0.0
            Boundary RMSE: 0.0
            Interface RMSE: 0.0
            Fourier
                - Low: 0.0
                - Mid: 0.0
                - High: 0.0
```

# Pool Boiling SubCooled

## G-FNO

```
CUDA_VISIBLE_DEVICES=6 python train.py data_base_dir=/nvme-data/jacob/BubbleML log_dir=/mnt/data/shared/jacob/GFNO/BubbleML/logs/PB_Subcooled/gfno/full_modes_pad_upscale dataset=PB_SubCooled experiment=gfno/pb_temp_full
data_base_dir: /nvme-data/jacob/BubbleML
log_dir: /mnt/data/shared/jacob/GFNO/BubbleML/logs/PB_Subcooled/gfno/full_modes_pad_upscale
train: true
test: true
model_checkpoint: null
dataset:
  name: subcooled
  transform: true
  steady_time: 30
  train_paths:
  - ${data_base_dir}/PoolBoiling-SubCooled-FC72-2D/Twall-79.hdf5
  - ${data_base_dir}/PoolBoiling-SubCooled-FC72-2D/Twall-81.hdf5
  - ${data_base_dir}/PoolBoiling-SubCooled-FC72-2D/Twall-85.hdf5
  - ${data_base_dir}/PoolBoiling-SubCooled-FC72-2D/Twall-90.hdf5
  - ${data_base_dir}/PoolBoiling-SubCooled-FC72-2D/Twall-95.hdf5
  - ${data_base_dir}/PoolBoiling-SubCooled-FC72-2D/Twall-98.hdf5
  - ${data_base_dir}/PoolBoiling-SubCooled-FC72-2D/Twall-103.hdf5
  - ${data_base_dir}/PoolBoiling-SubCooled-FC72-2D/Twall-106.hdf5
  - ${data_base_dir}/PoolBoiling-SubCooled-FC72-2D/Twall-110.hdf5
  val_paths:
  - ${data_base_dir}/PoolBoiling-SubCooled-FC72-2D/Twall-100.hdf5
experiment:
  torch_dataset_name: temp_input_dataset
  distributed: false
    train:
    max_epochs: 250
    batch_size: 8
    shuffle_data: true
    time_window: 5
    future_window: 5
    push_forward_steps: 1
    use_coords: true
    noise: true
    downsample_factor: 1
  model:
    model_name: gfno
    fmode_frac:
    - 0.165
    - 0.165
    width: 32
    reflection: false
    domain_padding: 0.1
  optimizer:
    initial_lr: 0.001
    weight_decay: 1.0e-06
  lr_scheduler:
    name: step
    patience: 75
    factor: 0.5

['${data_base_dir}/PoolBoiling-SubCooled-FC72-2D/Twall-79.hdf5', '${data_base_dir}/PoolBoiling-SubCooled-FC72-2D/Twall-81.hdf5', '${data_base_dir}/PoolBoiling-SubCooled-F
C72-2D/Twall-85.hdf5', '${data_base_dir}/PoolBoiling-SubCooled-FC72-2D/Twall-90.hdf5', '${data_base_dir}/PoolBoiling-SubCooled-FC72-2D/Twall-95.hdf5', '${data_base_dir}/PoolBoiling-SubCooled-FC72-2D/Twall-98.hdf5', '${data_base_dir}/PoolBoiling-SubCooled-FC72-2D/Twall-103.hdf5', '${data_base_dir}/PoolBoiling-SubCooled-FC72-2D/Twall-106.hdf5', '${data_base_dir}/PoolBoiling-SubCooled-FC72-2D/Twall-110.hdf5']
wall temp tensor(98.8603, dtype=torch.float64)
train size:  183
# parameters: 61032453
GFNO2d(
  (domain_padding): DomainPadding()
  (p): GConv2d()
  (conv0): GSpectralConv2d(
    (conv): GConv2d(
      (W): ParameterDict(
          (00_modes): Parameter containing: [torch.cuda.FloatTensor of size 32x1x32x4x1x1 (GPU 0)]
          (y0_modes): Parameter containing: [torch.cuda.ComplexFloatTensor of size 32x1x32x4x30x1 (GPU 0)]
          (yposx_modes): Parameter containing: [torch.cuda.ComplexFloatTensor of size 32x1x32x4x61x30 (GPU 0)]
      )
    )
  )
  (conv1): GSpectralConv2d(
    (conv): GConv2d(
      (W): ParameterDict(
          (00_modes): Parameter containing: [torch.cuda.FloatTensor of size 32x1x32x4x1x1 (GPU 0)]
          (y0_modes): Parameter containing: [torch.cuda.ComplexFloatTensor of size 32x1x32x4x30x1 (GPU 0)]
          (yposx_modes): Parameter containing: [torch.cuda.ComplexFloatTensor of size 32x1x32x4x61x30 (GPU 0)]
      )
    )
  )
  (conv2): GSpectralConv2d(
    (conv): GConv2d(
      (W): ParameterDict(
          (00_modes): Parameter containing: [torch.cuda.FloatTensor of size 32x1x32x4x1x1 (GPU 0)]
          (y0_modes): Parameter containing: [torch.cuda.ComplexFloatTensor of size 32x1x32x4x30x1 (GPU 0)]
          (yposx_modes): Parameter containing: [torch.cuda.ComplexFloatTensor of size 32x1x32x4x61x30 (GPU 0)]
      )
    )
  )
  (conv3): GSpectralConv2d(
    (conv): GConv2d(
      (W): ParameterDict(
          (00_modes): Parameter containing: [torch.cuda.FloatTensor of size 32x1x32x4x1x1 (GPU 0)]
          (y0_modes): Parameter containing: [torch.cuda.ComplexFloatTensor of size 32x1x32x4x30x1 (GPU 0)]
          (yposx_modes): Parameter containing: [torch.cuda.ComplexFloatTensor of size 32x1x32x4x61x30 (GPU 0)]
      )
    )
  )
  (mlp0): GMLP2d(
    (mlp1): GConv2d()
    (mlp2): GConv2d()
  )
  (mlp1): GMLP2d(
    (mlp1): GConv2d()
        (mlp2): GConv2d()
  )
  (mlp2): GMLP2d(
    (mlp1): GConv2d()
    (mlp2): GConv2d()
  )
  (mlp3): GMLP2d(
    (mlp1): GConv2d()
    (mlp2): GConv2d()
  )
  (w0): GConv2d()
  (w1): GConv2d()
  (w2): GConv2d()
  (w3): GConv2d()
  (norm): GNorm(
    (norm): InstanceNorm3d(32, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
  )
  (q): GMLP2d(
    (mlp1): GConv2d()
    (mlp2): GConv2d()
  )
)
Model has 30558213 parameters
Model has 61032453 parameters
<op_lib.temp_trainer.TempTrainer object at 0x7f3078253850>
epoch  0
...
saving model to /mnt/data/shared/jacob/GFNO/BubbleML/logs/PB_Subcooled/gfno/full_modes_pad_upscale/subcooled/GFNO2d_subcooled_temp_input_dataset_1699023826.pt
rollout time 2.1377575397491455 (s)
tensor(1.) tensor(-1.)
tensor(0.8168) tensor(-1.0030)
384 384
torch.Size([165, 1536])

            MAE: 0.025107556954026222
            RMSE: 0.06447415053844452
            Relative Error: 0.06673450022935867
            Max Error: 2.440495014190674
            Boundary RMSE: 0.15009145438671112
            Interface RMSE: 0.25941699743270874
            Fourier
                - Low: 0.5451402068138123
                - Mid: 0.5243523716926575
                - High: 0.06382591277360916

rollout time 1.9590654373168945 (s)
tensor(1.) tensor(-1.)
tensor(1.) tensor(-1.)
384 384
torch.Size([165, 1536])

            MAE: 0.0
            RMSE: 0.0
            Relative Error: 0.0
            Max Error: 0.0
            Boundary RMSE: 0.0
            Interface RMSE: 0.0
            Fourier
                - Low: 0.0
                - Mid: 0.0
                - High: 0.0


            MAE: 0.0
            RMSE: 0.0
            Relative Error: 0.0
            Max Error: 0.0
            Boundary RMSE: 0.0
            Interface RMSE: 0.0
            Fourier
                - Low: 0.0
                - Mid: 0.0
                - High: 0.0
```

## FNO
```
CUDA_VISIBLE_DEVICES=3 python train.py data_base_dir=/data/jacob/BubbleML log_dir=/mnt/data/shared/jacob/GFNO/BubbleML/logs/PB_Subcooled/fno_upscale/ dataset=PB_SubCooled experiment=fno/pb_temp_full
data_base_dir: /data/jacob/BubbleML
log_dir: /mnt/data/shared/jacob/GFNO/BubbleML/logs/PB_Subcooled/fno_upscale/
train: true
test: true
model_checkpoint: null
dataset:
  name: subcooled
  transform: true
  steady_time: 30
  train_paths:
  - ${data_base_dir}/PoolBoiling-SubCooled-FC72-2D/Twall-79.hdf5
  - ${data_base_dir}/PoolBoiling-SubCooled-FC72-2D/Twall-81.hdf5
  - ${data_base_dir}/PoolBoiling-SubCooled-FC72-2D/Twall-85.hdf5
  - ${data_base_dir}/PoolBoiling-SubCooled-FC72-2D/Twall-90.hdf5
  - ${data_base_dir}/PoolBoiling-SubCooled-FC72-2D/Twall-95.hdf5
  - ${data_base_dir}/PoolBoiling-SubCooled-FC72-2D/Twall-98.hdf5
  - ${data_base_dir}/PoolBoiling-SubCooled-FC72-2D/Twall-103.hdf5
  - ${data_base_dir}/PoolBoiling-SubCooled-FC72-2D/Twall-106.hdf5
  - ${data_base_dir}/PoolBoiling-SubCooled-FC72-2D/Twall-110.hdf5
  val_paths:
  - ${data_base_dir}/PoolBoiling-SubCooled-FC72-2D/Twall-100.hdf5
experiment:
  torch_dataset_name: temp_input_dataset
  distributed: false
  train:
    max_epochs: 250
    batch_size: 8
    shuffle_data: true
    time_window: 5
    future_window: 5
    push_forward_steps: 1
    use_coords: true
    noise: true
    downsample_factor: 1
  model:
    model_name: fno
    fmode_frac:
    - 0.165
    - 0.165
    hidden_channels: 64
    domain_padding:
       - 0.1
    - 0.1
    n_layers: 4
    norm: group_norm
  optimizer:
    initial_lr: 0.001
    weight_decay: 1.0e-06
  lr_scheduler:
    name: step
    patience: 75
    factor: 0.5

['${data_base_dir}/PoolBoiling-SubCooled-FC72-2D/Twall-79.hdf5', '${data_base_dir}/PoolBoiling-SubCooled-FC72-2D/Twall-81.hdf5', '${data_base_dir}/PoolBoiling-SubCooled-FC72-2D/Twall-85.hdf5', '${data_base_dir}/PoolBoiling-SubCooled-FC72-2D/Twall-90.hdf5', '${data_base_dir}/PoolBoiling-SubCooled-FC72-2D/Twall-95.hdf5', '${data_base_dir}/PoolBoiling-SubCooled-FC72-2D/Twall-98.hdf5', '${data_base_dir}/PoolBoiling-SubCooled-FC72-2D/Twall-103.hdf5', '${data_base_dir}/PoolBoiling-SubCooled-FC72-2D/Twall-106.hdf5', '${data_base_dir}/PoolBoiling-SubCooled-FC72-2D/Twall-110.hdf5']
['${data_base_dir}/PoolBoiling-SubCooled-FC72-2D/Twall-79.hdf5', '${data_base_dir}/PoolBoiling-SubCooled-FC72-2D/Twall-81.hdf5', '${data_base_dir}/PoolBoiling-SubCooled-FC72-2D/Twall-85.hdf5', '${data_base_dir}/PoolBoiling-SubCooled-FC72-2D/Twall-90.hdf5', '${data_base_dir}/PoolBoiling-SubCooled-FC72-2D/Twall-95.hdf5', '${data_base_dir}/PoolBoiling-SubCooled-FC72-2D/Twall-98.hdf5', '${data_base_dir}/PoolBoiling-SubCooled-FC72-2D/Twall-103.hdf5', '${data_base_dir}/PoolBoiling-SubCooled-FC72-2D/Twall-106.hdf5', '${data_base_dir}/PoolBoiling-SubCooled-FC72-2D/Twall-110.hdf5']
wall temp tensor(98.8603, dtype=torch.float64)
train size:  183
FNO(
  (domain_padding): DomainPadding()
  (fno_blocks): FNOBlocks(
    (convs): FactorizedSpectralConv(
      (weight): ModuleList(
        (0): ComplexDenseTensor(shape=torch.Size([64, 64, 31, 31]), rank=None)
        (1): ComplexDenseTensor(shape=torch.Size([64, 64, 31, 31]), rank=None)
        (2): ComplexDenseTensor(shape=torch.Size([64, 64, 31, 31]), rank=None)
        (3): ComplexDenseTensor(shape=torch.Size([64, 64, 31, 31]), rank=None)
        (4): ComplexDenseTensor(shape=torch.Size([64, 64, 31, 31]), rank=None)
        (5): ComplexDenseTensor(shape=torch.Size([64, 64, 31, 31]), rank=None)
        (6): ComplexDenseTensor(shape=torch.Size([64, 64, 31, 31]), rank=None)
        (7): ComplexDenseTensor(shape=torch.Size([64, 64, 31, 31]), rank=None)
      )
    )
    (fno_skips): ModuleList(
      (0): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (2): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (3): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
    )
    (norm): ModuleList(
      (0): GroupNorm(1, 64, eps=1e-05, affine=True)
      (1): GroupNorm(1, 64, eps=1e-05, affine=True)
      (2): GroupNorm(1, 64, eps=1e-05, affine=True)
      (3): GroupNorm(1, 64, eps=1e-05, affine=True)
    )
  )
  (lifting): Lifting(
    (fc): Conv2d(27, 64, kernel_size=(1, 1), stride=(1, 1))
  )
  (projection): Projection(
    (fc1): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1))
    (fc2): Conv2d(256, 5, kernel_size=(1, 1), stride=(1, 1))
  )
)
Model has 63016965 parameters
Model has 63016965 parameters
<op_lib.temp_trainer.TempTrainer object at 0x7f3e7df0d820>
epoch  0
...
saving model to /mnt/data/shared/jacob/GFNO/BubbleML/logs/PB_Subcooled/fno_upscale/subcooled/FNO_subcooled_temp_input_dataset_1699048481.pt
rollout time 2.2548537254333496 (s)
tensor(0.8763) tensor(-1.)
tensor(0.8168) tensor(-1.0030)
384 384
torch.Size([165, 1536])

            MAE: 0.028747260570526123
            RMSE: 0.07190752774477005
            Relative Error: 0.0744137093424797
            Max Error: 2.3808577060699463
            Boundary RMSE: 0.16091684997081757
            Interface RMSE: 0.24930493533611298
            Fourier
                - Low: 0.6750268936157227
                - Mid: 0.5794897079467773
                - High: 0.06581910699605942

rollout time 2.5190670490264893 (s)
tensor(0.8763) tensor(-1.)
tensor(0.8763) tensor(-1.)
384 384
torch.Size([165, 1536])

            MAE: 0.0
            RMSE: 0.0
            Relative Error: 0.0
            Max Error: 0.0
            Boundary RMSE: 0.0
            Interface RMSE: 0.0
            Fourier
                - Low: 0.0
                - Mid: 0.0
                - High: 0.0


            MAE: 0.0
            RMSE: 0.0
            Relative Error: 0.0
            Max Error: 0.0
            Boundary RMSE: 0.0
            Interface RMSE: 0.0
            Fourier
                - Low: 0.0
                - Mid: 0.0
                - High: 0.0
```
