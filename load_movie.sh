nohup python -u new_data.py \
  --data=Movie \
  --job=10 \
  --item_max_length=50 \
  --user_max_length=50 \
  --k_hop=3 \
  > ./logs/movie_preprocess.log 2>&1 &
