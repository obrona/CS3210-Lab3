export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
alias tv100="srun --gpus=1 --constraint=xgpc"
alias tvolta="srun --gpus=1 --constraint=xgpd"
alias trtx="srun --gpus=1 --constraint=xgpe"
alias tt4="srun --gpus=1 --constraint=xgpf"
alias a100mig="srun --gpus=a100-40 --constraint=xgph"
alias a100a="srun --gpus=a100-40 --constraint=xgpg"
alias a100b="srun --gpus=a100-80"
alias h100mig="srun --gpus=h100-47"
alias h100="srun --gpus=h100-96"