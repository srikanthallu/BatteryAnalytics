python main_nbeats.py --gpus 3 --epochs 10000 --save_freq 1000 \
--stacks 5 --loss RMSE --dir_out results/local_data/ 

#interpretable
python main_nbeats.py --gpus 3 --epochs 10000 --model_type interpretable \
--save_freq 500 --stacks 10 --harmonic 4 --loss RMSE --dir_out results/seasonal/ 

#test interpretable
python test.py --model_type interpretable \
--stacks 10 --harmonic 4 --loss RMSE --dir_out results/seasonal/epoch1000/ \
--dir_model results/seasonal/ckpt_epoch1000.pt