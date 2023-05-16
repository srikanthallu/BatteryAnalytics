$dir_home = /home/oit/electrode-micro-ml/sem-seg/bharat-result
python train_idk.py --device gpu \
  --gpus 3 --epochs 100 \
  --batch_size 20000 \
  --save_freq 10 \
  --learning_rate 0.0001 \
  --dir_img /home/oit/electrode-micro-ml/sem-seg/bharat-result/battery_2/train_images/ \
  --dir_train /home/oit/electrode-micro-ml/sem-seg/bharat-result/output/m1-idk-predict-train/ \
  --dir_test /home/oit/electrode-micro-ml/sem-seg/bharat-result/output/m1-idk-predict-val/ \
  --dir_misclass misclass/ \
  --dir_m1out prediction/ \
  --dir_uq uq/ \
  --model_path /home/oit/electrode-micro-ml/sem-seg/bharat-result/checkpoint/idk/ \
  --dir_out /home/oit/electrode-micro-ml/sem-seg/bharat-result/output/idk-train/ 


#predict IDK
#change model path ckpt_epoch accordingly
#change dir_test m1-idk-predict-val/m1-idk-predict-test
  python train_idk.py --device gpu \
  --test_only \
  --batch_size 20000 \
  --dir_img /home/oit/electrode-micro-ml/sem-seg/bharat-result/battery_2/test_images/ \
  --dir_train /home/oit/electrode-micro-ml/sem-seg/bharat-result/output/m1-idk-predict-train/ \
  --dir_test /home/oit/electrode-micro-ml/sem-seg/bharat-result/output/m1-idk-predict-test/ \
  --dir_misclass misclass/ \
  --dir_m1out prediction/ \
  --dir_uq uq/ \
  --model_path /home/oit/electrode-micro-ml/sem-seg/bharat-result/checkpoint/idk/last.pth \
  --dir_out /home/oit/electrode-micro-ml/sem-seg/bharat-result/output/idk-test/

  #train local model M2
python main_m2_ce.py --gpus 3 \
--batch_size 2048 \
--loc_size 5 \
--in_channel 4 \
--epochs 100 \
--num_workers 8 \
--learning_rate 0.001 \
--print_freq 10 \
--save_freq 3 \
--dir_img /home/oit/electrode-micro-ml/sem-seg/bharat-result/battery_2/ \
--dir_mask /home/oit/electrode-micro-ml/sem-seg/bharat-result/battery_2/ \
--dir_idk_train /home/oit/electrode-micro-ml/sem-seg/bharat-result/output/idk-train/ \
--dir_idk_test /home/oit/electrode-micro-ml/sem-seg/bharat-result/output/idk-val/ \
--dir_uq_train /home/oit/electrode-micro-ml/sem-seg/bharat-result/output/m1-idk-predict-train/uq/ \
--dir_uq_test /home/oit/electrode-micro-ml/sem-seg/bharat-result/output/m1-idk-predict-val/uq/ \
--dir_m1out_train /home/oit/electrode-micro-ml/sem-seg/bharat-result/output/m1-idk-predict-train/prediction/ \
--dir_m1out_test /home/oit/electrode-micro-ml/sem-seg/bharat-result/output/m1-idk-predict-val/prediction/ \
--model_path /home/oit/electrode-micro-ml/sem-seg/bharat-result/checkpoint/m2/ \
--dir_out /home/oit/electrode-micro-ml/sem-seg/bharat-result/output/m2/

#inference local model M2
python m2_predict.py --device cpu \
--batch_size 2048 \
--test_only \
--loc_size 5 \
--in_channel 4 \
--dir_img /home/oit/electrode-micro-ml/sem-seg/bharat-result/battery_2/test_images/ \
--dir_mask /home/oit/electrode-micro-ml/sem-seg/bharat-result/battery_2/test_label/ \
--dir_pretrain_m2 /home/oit/electrode-micro-ml/sem-seg/bharat-result/checkpoint/m2/ckpt_epoch12.pth \
--dir_idk /home/oit/electrode-micro-ml/sem-seg/bharat-result/output/idk-test/ \
--dir_outm1 /home/oit/electrode-micro-ml/sem-seg/bharat-result/output/m1-idk-predict-test/prediction/ \
--dir_m1 /home/oit/electrode-micro-ml/sem-seg/bharat-result/output/m1-idk-predict-test/prediction/ \
--dir_out /home/oit/electrode-micro-ml/sem-seg/bharat-result/output/m2-test/

python evaluation.py --topk 5 --num_class 3 --exp chemphase \
--dir_gt /home/oit/electrode-micro-ml/sem-seg/bharat-result/battery_2/test_label/ \
--dir_pred /home/oit/electrode-micro-ml/sem-seg/bharat-result/output/m2-test/ensemble/ \
--dir_out /home/oit/electrode-micro-ml/sem-seg/bharat-result/output/m2-test/ensemble/ 


