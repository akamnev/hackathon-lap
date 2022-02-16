echo "specify device"
read device
python train_v2.py \
  --exp_name=mdl-L4-H32-Pmean-NF-BF \
  --seed=3088 \
  --dev_reverse=1 \
  --start_epoch=20000 \
  --num_epoch=5000 \
  --device="$device" \
  --batch_size=256
