echo "specify device"
read device
num_epoch=2000
python search_v1.py \
  --exp_name=L4-H32-Patt-NF-BF \
  --device="$device" \
  --batch_size=256 \
  --num_epoch=$num_epoch \
