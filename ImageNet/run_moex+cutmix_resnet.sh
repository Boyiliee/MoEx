# Note: Please adjust the batch size according to your GPU memory. For example, you may use -b 128 or -b 64. The learning rate is adjusted based on the batch size.
DATA='YOUR_IMAGENET_PATH'
NGPUS=8  #number of GPUs to use
WORKERS_PER_GPU=10  #the number of workers to pre-process the data for each GPU
batch_size=224
norm_method=pono
arch=resnet50
epochs=300
lam=0.9
prob=0.25

python -m torch.distributed.launch --nproc_per_node=${NGPUS} \
main_moex_cutmix.py -a moex_${arch} -b ${batch_size} --workers ${WORKERS_PER_GPU} \
--opt-level O1 --output_dir save/${arch}_moex+cutmix --min_lr 1e-9 --lr_scheduler cosine \
--epochs ${epochs} ${DATA} --print-freq 20 --img-size 224 --beta 1.0 --cutmix_prob 1 \
--moex_norm ${norm_method} --moex_lam ${lam} --moex_prob ${prob}