## requirements:
### Computer environment
```
Anaconda
cuda&cudnn
```
### creating venv
```Anaconda
conda create -n SAC python==3.8.13
conda activate SAC
pip install -r requirements.txt
pip install gym[classic_control]
pip install gym[gym]
```

## test using the pre-train Model(Results)
```python
python SAC_V0_bw3_v1 --test
python SAC_V0_bw3 --test
python SAC_V0 --test
```

## train using the pre-train Model
```python
python SAC_V0_bw3_v1 --train
python SAC_V0_bw3 --train
python SAC_V0 --train
```

## load the previous model & train it
```python
python SAC_V0_bw3_v1 --train --load
python SAC_V0_bw3 --train --load
python SAC_V0 --train --load
```

## Results
![open the tensorboardX]("pic/Result.png")
