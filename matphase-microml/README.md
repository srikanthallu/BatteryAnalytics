We have three model: run each one sequentially
Python Scripts:
1. Train Global model: main_m1.py
2. Get Global model output for pretrained model: m1_idk_predict.py
3. Train IDK: idk_train.py
4. predict IDK: idk_train.py
5. Train M2: main_m2_ce.py
6. test M2: m2_predict.py
7. evaluation.py: compute paper evalaution metrics
8. metric.py: compute F1-score

All pretrained models: checkpoint/

-----An example script to run all the models---------
shell-scripts/run.sh

----Example scripts to train M2-------
python main_m2_ce.py --gpus 4 batch_size 1024 loc_size 5 in_channel 1 epochs 10 num_workers 8 learning_rate 0.001 print_freq 10 save_freq 3 dir_img ../data/ dir_mask ../data/ dir_idk_train ../output/idk/train/ dir_idk_test ../output/idk/validation/ dir_m1out_train ../output/m1/train/prediction/ dir_m1out_test ../output/m1/validation/prediction/ model_path checkpoint/m2-in_1-embed/ dir_out ../output/m2-in_1-embed/

Required parameters:
a. dir_img: input image data
b. dir_mask: GT label data
c. dir_idk_train: directory for idk instances for training data
d. dir_idk_test: directory for idk instances for test/validation data
e. dir_m1out_train: directory for m1 logits for training data
f. dir_m1out_test: directory for m1 logits for test/validation data
g. model_path: model directory to save
h. dir_out: output directory

----Example scripts to test M2 for final prediction-------
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

Required parameters:
a. dir_img: input image data
b. dir_mask: GT label data
c. dir_idk: directory for idk instances for training data
d. loc_size: input local region size
e. dir_outm1: directory for M1 logits
f. dir_m1: directory for m1 predictions
g. dir_out: output directory to save:
	i. Ensemble/: original ensemble predictions , i.e., final output (both .npy, .png format)
	ii. misclass/: misclass image, where final output misclass
	iii. f1_ensemble: f1 score for output
	iv. f1_m2_cnn: f1 score for m2 model


