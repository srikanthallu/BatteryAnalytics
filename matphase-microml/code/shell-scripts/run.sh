#!/bin/sh
#train global model M1
python main_m1.py --gpus 4 --epochs 200 \
--batch_size 22 --idk_alpha 0.0001 
--dir_img ../data/ \
--dir_mask ../data/ \
--dir_checkpoint checkpoint/m1/

#predict global model M1 
python m1_idk_predict.py --idk_th 0.01 --device gpu \
--test_only 
--dir_img ../data/test_images/ \
--dir_mask ../data/test_label/ \
--dir_pretrain checkpoint/m1/ckpt_epoch100.pth \
--dir_out ../output/m1-idk/

#train IDK 
  python train_idk.py --device gpu \
  --gpus 4 --epochs 100 \
  --batch_size 20000 \
  --save_freq 10 \
  --learning_rate 0.0001 \
  --dir_img ../data/train_images/ \
  --dir_train ../output/m1/train/ \
  --dir_test ../output/m1/validation/ \
  --dir_misclass misclass/ \
  --dir_m1out prediction/ \
  --dir_uq uq/ \
  --model_path checkpoint/idk/ \
  --dir_out ../output/idk/train/ 

#predict IDK
  python train_idk.py --device gpu \
  --test_only \
  --batch_size 10000 \
  --dir_img ../data/validation_images/ \
  --dir_train ../output/m1/train/ \
  --dir_test ../output/m1/validation/ \
  --dir_misclass misclass/ \
  --dir_m1out prediction/ \
  --dir_uq uq/ \
  --model_path checkpoint/idk/ckpt_epoch60.pth \
  --dir_out ../output/idk/validation/ 

#train local model M2
python main_m2_ce.py --gpus 4 \
--batch_size 1024 \
--loc_size 5 \
--in_channel 1 \
--epochs 10 \
--num_workers 8 \
--learning_rate 0.001 \
--print_freq 10 \
--save_freq 3 \
--dir_img ../data/ \
--dir_mask ../data/ \
--dir_idk_train ../output/idk/train/ \
--dir_idk_test ../output/idk/validation/ \
--dir_m1out_train ../output/m1/train/prediction/ \
--dir_m1out_test ../output/m1/validation/prediction/ \
--model_path checkpoint/m2-in_1-embed/ \
--dir_out ../output/m2-in_1-embed/

#predict local model M2
python m2_predict.py --device cpu \
--batch_size 1024 \
--test_only \
--loc_size 5 \
--dir_img ../data/test_images/ \
--dir_mask ../data/test_label/ \
--dir_pretrain_m2 checkpoint/m2/ckpt_epoch20.pth \
--dir_idk ../output/idk/test/ \
--dir_outm1 ../output/m1/test/prediction/ \
--dir_m1 ../output/m1/test/prediction/ \
--dir_out ../output/m2-cnn/

#train IDK 
  python train_idk.py --device gpu \
  --gpus 4 --epochs 100 \
  --batch_size 20000 \
  --save_freq 10 \
  --learning_rate 0.0001 \
  --dir_img ../data/train_images/ \
  --dir_train ../output/m1/train/ \
  --dir_test ../output/m1/validation/ \
  --dir_misclass misclass/ \
  --dir_m1out prediction/ \
  --dir_uq uq/ \
  --model_path checkpoint/idk/ \
  --dir_out ../output/idk/train/ 

#predict IDK
  python train_idk.py --device gpu \
  --test_only \
  --batch_size 10000 \
  --dir_img ../data/validation_images/ \
  --dir_train ../output/m1/train/ \
  --dir_test ../output/m1/validation/ \
  --dir_misclass misclass/ \
  --dir_m1out prediction/ \
  --dir_uq uq/ \
  --model_path checkpoint/idk/ckpt_epoch60.pth \
  --dir_out ../output/idk/validation/ 


