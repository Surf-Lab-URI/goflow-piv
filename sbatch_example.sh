#!/bin/bash
#SBATCH --job-name="dns_trans_test"
#SBATCH --output="dns_trans_test.%j.%N.out"
#SBATCH --partition=gpu-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --mem=96000M
#SBATCH --account=uri107
#SBATCH --no-requeue
#SBATCH -t 18:00:00

# submit this job using the following command: sbatch sbatch_example.sh

cd /home/agoering/agoering/goflow-piv
source activate base
conda activate goflow-piv


python train_goflow.py --data_root /home/agoering/agoering/cai2018_dataset/DNS_turbulence --cuda 0 --model unet --nbase 16 --epochs 100 --rand_trans 0 --batch_size 64 --lr 0.001 --eval_criterion r2 --write_log --output_dir ./dns_trans_test/
python train_goflow.py --data_root /home/agoering/agoering/cai2018_dataset/DNS_turbulence --cuda 0 --model unet --nbase 16 --epochs 100 --rand_trans 0 --batch_size 64 --lr 0.001 --eval_criterion r2 --write_log --output_dir ./dns_trans_test/ --use_grad_loss --c_spec 0.5 --resume

python train_goflow.py --data_root /home/agoering/agoering/cai2018_dataset/DNS_turbulence --cuda 0 --model unet --nbase 16 --epochs 100 --rand_trans 1 --batch_size 64 --lr 0.001 --eval_criterion r2 --write_log --output_dir ./dns_trans_test/
python train_goflow.py --data_root /home/agoering/agoering/cai2018_dataset/DNS_turbulence --cuda 0 --model unet --nbase 16 --epochs 100 --rand_trans 1 --batch_size 64 --lr 0.001 --eval_criterion r2 --write_log --output_dir ./dns_trans_test/ --use_grad_loss --c_spec 0.5 --resume


python train_goflow.py --data_root /home/agoering/agoering/cai2018_dataset/DNS_turbulence --cuda 0 --model unet --nbase 16 --epochs 100 --rand_trans 2 --batch_size 64 --lr 0.001 --eval_criterion r2 --write_log --output_dir ./dns_trans_test/
python train_goflow.py --data_root /home/agoering/agoering/cai2018_dataset/DNS_turbulence --cuda 0 --model unet --nbase 16 --epochs 100 --rand_trans 2 --batch_size 64 --lr 0.001 --eval_criterion r2 --write_log --output_dir ./dns_trans_test/ --use_grad_loss --c_spec 0.5 --resume


python train_goflow.py --data_root /home/agoering/agoering/cai2018_dataset/DNS_turbulence --cuda 0 --model unet --nbase 16 --epochs 100 --rand_trans 3 --batch_size 64 --lr 0.001 --eval_criterion r2 --write_log --output_dir ./dns_trans_test/
python train_goflow.py --data_root /home/agoering/agoering/cai2018_dataset/DNS_turbulence --cuda 0 --model unet --nbase 16 --epochs 100 --rand_trans 3 --batch_size 64 --lr 0.001 --eval_criterion r2 --write_log --output_dir ./dns_trans_test/ --use_grad_loss --c_spec 0.5 --resume


python train_goflow.py --data_root /home/agoering/agoering/cai2018_dataset/DNS_turbulence --cuda 0 --model unet --nbase 16 --epochs 100 --rand_trans 5 --batch_size 64 --lr 0.001 --eval_criterion r2 --write_log --output_dir ./dns_trans_test/
python train_goflow.py --data_root /home/agoering/agoering/cai2018_dataset/DNS_turbulence --cuda 0 --model unet --nbase 16 --epochs 100 --rand_trans 5 --batch_size 64 --lr 0.001 --eval_criterion r2 --write_log --output_dir ./dns_trans_test/ --use_grad_loss --c_spec 0.5 --resume


python train_goflow.py --data_root /home/agoering/agoering/cai2018_dataset/DNS_turbulence --cuda 0 --model unet --nbase 16 --epochs 100 --rand_trans 7 --batch_size 64 --lr 0.001 --eval_criterion r2 --write_log --output_dir ./dns_trans_test/
python train_goflow.py --data_root /home/agoering/agoering/cai2018_dataset/DNS_turbulence --cuda 0 --model unet --nbase 16 --epochs 100 --rand_trans 7 --batch_size 64 --lr 0.001 --eval_criterion r2 --write_log --output_dir ./dns_trans_test/ --use_grad_loss --c_spec 0.5 --resume


python train_goflow.py --data_root /home/agoering/agoering/cai2018_dataset/DNS_turbulence --cuda 0 --model unet --nbase 16 --epochs 100 --rand_trans 10 --batch_size 64 --lr 0.001 --eval_criterion r2 --write_log --output_dir ./dns_trans_test/
python train_goflow.py --data_root /home/agoering/agoering/cai2018_dataset/DNS_turbulence --cuda 0 --model unet --nbase 16 --epochs 100 --rand_trans 10 --batch_size 64 --lr 0.001 --eval_criterion r2 --write_log --output_dir ./dns_trans_test/ --use_grad_loss --c_spec 0.5 --resume


python train_goflow.py --data_root /home/agoering/agoering/cai2018_dataset/DNS_turbulence --cuda 0 --model unet --nbase 16 --epochs 100 --rand_trans 13 --batch_size 64 --lr 0.001 --eval_criterion r2 --write_log --output_dir ./dns_trans_test/
python train_goflow.py --data_root /home/agoering/agoering/cai2018_dataset/DNS_turbulence --cuda 0 --model unet --nbase 16 --epochs 100 --rand_trans 13 --batch_size 64 --lr 0.001 --eval_criterion r2 --write_log --output_dir ./dns_trans_test/ --use_grad_loss --c_spec 0.5 --resume


python train_goflow.py --data_root /home/agoering/agoering/cai2018_dataset/DNS_turbulence --cuda 0 --model unet --nbase 16 --epochs 100 --rand_trans 16 --batch_size 64 --lr 0.001 --eval_criterion r2 --write_log --output_dir ./dns_trans_test/
python train_goflow.py --data_root /home/agoering/agoering/cai2018_dataset/DNS_turbulence --cuda 0 --model unet --nbase 16 --epochs 100 --rand_trans 16 --batch_size 64 --lr 0.001 --eval_criterion r2 --write_log --output_dir ./dns_trans_test/ --use_grad_loss --c_spec 0.5 --resume


python train_goflow.py --data_root /home/agoering/agoering/cai2018_dataset/DNS_turbulence --cuda 0 --model unet --nbase 16 --epochs 100 --rand_trans 20 --batch_size 64 --lr 0.001 --eval_criterion r2 --write_log --output_dir ./dns_trans_test/
python train_goflow.py --data_root /home/agoering/agoering/cai2018_dataset/DNS_turbulence --cuda 0 --model unet --nbase 16 --epochs 100 --rand_trans 20 --batch_size 64 --lr 0.001 --eval_criterion r2 --write_log --output_dir ./dns_trans_test/ --use_grad_loss --c_spec 0.5 --resume


python train_goflow.py --data_root /home/agoering/agoering/cai2018_dataset/DNS_turbulence --cuda 0 --model unet --nbase 16 --epochs 100 --rand_trans 24 --batch_size 64 --lr 0.001 --eval_criterion r2 --write_log --output_dir ./dns_trans_test/
python train_goflow.py --data_root /home/agoering/agoering/cai2018_dataset/DNS_turbulence --cuda 0 --model unet --nbase 16 --epochs 100 --rand_trans 24 --batch_size 64 --lr 0.001 --eval_criterion r2 --write_log --output_dir ./dns_trans_test/ --use_grad_loss --c_spec 0.5 --resume


python train_goflow.py --data_root /home/agoering/agoering/cai2018_dataset/DNS_turbulence --cuda 0 --model unet --nbase 16 --epochs 100 --rand_trans 30 --batch_size 64 --lr 0.001 --eval_criterion r2 --write_log --output_dir ./dns_trans_test/
python train_goflow.py --data_root /home/agoering/agoering/cai2018_dataset/DNS_turbulence --cuda 0 --model unet --nbase 16 --epochs 100 --rand_trans 30 --batch_size 64 --lr 0.001 --eval_criterion r2 --write_log --output_dir ./dns_trans_test/ --use_grad_loss --c_spec 0.5 --resume

