#! /bin/bash
data=$1
batch=$2
ckpt=$3
exp_name=$4
out_path=$5

python predict.py --data $data/iid_test/Images \
                  --batch $batch \
                  --ckpt $ckpt \
                  --exp_name $exp_name --test_data iid

python predict.py --data $data/nuisances/shape/Images \
                  --batch $batch \
                  --ckpt $ckpt \
                  --exp_name $exp_name --test_data shape

python predict.py --data $data/nuisances/pose/Images \
                  --batch $batch \
                  --ckpt $ckpt \
                  --exp_name $exp_name --test_data pose

python predict.py --data $data/nuisances/context/Images \
                  --batch $batch \
                  --ckpt $ckpt \
                  --exp_name $exp_name --test_data context

python predict.py --data $data/nuisances/texture/Images \
                  --batch $batch \
                  --ckpt $ckpt \
                  --exp_name $exp_name --test_data texture

python predict.py --data $data/nuisances/occlusion/Images \
                  --batch $batch \
                  --ckpt $ckpt \
                  --exp_name $exp_name --test_data occlusion

python predict.py --data $data/nuisances/weather/Images \
                  --batch $batch \
                  --ckpt $ckpt \
                  --exp_name $exp_name --test_data weather

zip -j $out_path results/$exp_name/*.csv
mv $out_path results/$exp_name/
