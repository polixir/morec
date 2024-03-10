# MOREC (ICLR'24)
This repository is the official implementation of [Model-based Offline reinforcement learning with Reward Consistency (MOREC)](https://openreview.net/forum?id=GSBHKiw19c).
## Dependencies
### Install via pip
```bash
# logger
git clone https://github.com/FanmingL/SmartLogger
cd SmartLogger 
pip install -e .
cd ..
pip install gym==0.24.1 mujoco==2.3.6 mujoco-py==2.1.2.14 numpy==1.22.3
# other dependency
pip install torch tqdm 
# also should install NeoRL via pip install -e 
git clone https://github.com/polixir/NeoRL
cd NeoRL
pip install -e .
```

### docker

```bash
docker pull core.116.172.93.164.nip.io:30670/public/luofanming:20231013102110
```

## Pretrained Dynamics Model
The pretrained dynamics rewards and dynamics models are stored in [this url](https://box.nju.edu.cn/d/dd0ea2df0e4548f6bbdd/). 
Please unzip `d4rl_dataset.zip`, `dynamics_reward_models.tar.gz`, and  `learned_dynamics.tar.gz` to `MOREC/pretrained`.

We will release the dynamics reward learning code in a near future.

## Train
### MOREC-MOPO
```bash
# generate startup commands
python generate_tmuxp_morec_mopo.py
# execute the commands with tmuxp
tmuxp load run_all.json
```

### MOREC-MOBILE
```bash
# generate startup commands
python generate_tmuxp_morec_mobile.py
# execute the commands with tmuxp
tmuxp load run_all.json
```

## Acknowledgement
MOREC is built upon [OfflineRL-Kit](https://github.com/yihaosun1124/OfflineRL-Kit).

