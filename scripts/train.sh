echo "specify device"
read device
num_epoch=20000
for v in {1..5..1}
do
  seed=$RANDOM
  echo $seed
  python train_v1.py \
    --exp_name=mdl-L4-H32-Pmean-NF-BF \
    --device="$device" \
    --batch_size=256 \
    --num_epoch=$num_epoch \
    --seed=$seed \
    --dev_reverse=0
  python train_v1.py \
    --exp_name=mdl-L4-H32-Pmean-NF-BF \
    --device="$device" \
    --batch_size=256 \
    --num_epoch=$num_epoch \
    --seed=$seed \
    --dev_reverse=1
done