DATA=cifar100 #cifar10
JOB=pyramidnet_moex
epoch=300
lam=0.5
prob=0.25

python train_pono.py \
    --net_type pyramidnet_moex \
    --dataset ${DATA} \
    --depth 200 \
    --alpha 240 \
    --batch_size 64 \
    --lr 0.25 \
    --expname ${JOB} \
    --epochs ${epoch} \
    --beta 1.0 \
    --lam ${lam} --moex_prob ${prob}
