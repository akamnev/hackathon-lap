echo "specify device"
read device
num_epoch=20000
for v in {1..5..1}
do
  python train_v1.py \
    --exp_name=mdl-L4-H32-Pmean-NF-BF-try-"$v" \
    --device="$device" \
    --batch_size=256 \
    --num_epoch=$num_epoch \
    --seed=3088 \
    --dev_reverse=0
done