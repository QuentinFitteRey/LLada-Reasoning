# LLada-Reasoning

To set up the project in PACE, first create all folder and clone git hub:
```bash
cd scratch/
git clone https://github.com/QuentinFitteRey/LLada-Reasoning
mkdir envs
cd LLada-Reasoning/
```

Need to create the env, activate it and install required packages:
```bash
conda create -p ~/scratch/envs/llada python=3.11
conda activate ~/scratch/envs/llada
pip install -r requirements.txt
```

You should export your HF cache towards scratch to avoid.
```bash
export HF_HOME=~/scratch/envs/cache
```
Put it in your bashrc, it will avoid doing export HF everytime 

This contains only base packages such as transformers, torch, trl and accelerate. We might need others but currently this is sufficient.

