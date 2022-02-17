echo "specify device"
read device
for v in {1..5..1}
do
  seed=$RANDOM
  echo $seed
  python train_v4.py \
    --exp_name=b-L4-H32-Patt-NF-BF \
    --device="$device" \
    --seed=$seed \
    --dev_reverse=0
  python train_v4.py \
    --exp_name=b-L4-H32-Patt-NF-BF \
    --device="$device" \
    --seed=$seed \
    --dev_reverse=1
done