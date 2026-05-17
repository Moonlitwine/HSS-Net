#!/bin/bash

cfg=allweather.yaml
exp=exp_allweather
device=cuda:0

# python train.py --cfg=$cfg --exp=$exp --device=$device

test_resume=exp_allweather/ckpts/Allweather240000.pth
sample_folder=exp_allweather/sample_result/step240000

python test.py --cfg=$cfg --resume=$test_resume --sample_set=../datasets/RainDrop --sample_folder=$sample_folder --device=$device --not_save_result --calc_in_Y

# python test.py --cfg=$cfg --resume=$test_resume --sample_set=../datasets/Outdoor_Rain --sample_folder=$sample_folder --device=$device --not_save_result --calc_in_Y

# python test.py --cfg=$cfg --resume=$test_resume --sample_set=../datasets/Snow100K-L --sample_folder=$sample_folder --device=$device --not_save_result --calc_in_Y

# python test.py --cfg=$cfg --resume=$test_resume --sample_set=../datasets/Snow100K-S --sample_folder=$sample_folder --device=$device --not_save_result --calc_in_Y