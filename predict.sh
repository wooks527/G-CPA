#! /bin/bash
batch=$1
ckpt=$2
exp_name=$3
out_path=$4

python predict.py --data /data1/hyunwook/data/ood-cv/iid_test/ \
                  --batch $batch \
                  --ckpt $ckpt \
                  --exp_name $exp_name --test_data iid

python predict.py --data /data1/hyunwook/data/ood-cv/context/ \
                  --batch $batch \
                  --ckpt $ckpt \
                  --exp_name $exp_name --test_data context

python predict.py --data /data1/hyunwook/data/ood-cv/occlusion/ \
                  --batch $batch \
                  --ckpt $ckpt \
                  --exp_name $exp_name --test_data occlusion

python predict.py --data /data1/hyunwook/data/ood-cv/pose/ \
                  --batch $batch \
                  --ckpt $ckpt \
                  --exp_name $exp_name --test_data pose

python predict.py --data /data1/hyunwook/data/ood-cv/shape/ \
                  --batch $batch \
                  --ckpt $ckpt \
                  --exp_name $exp_name --test_data shape

python predict.py --data /data1/hyunwook/data/ood-cv/texture/ \
                  --batch $batch \
                  --ckpt $ckpt \
                  --exp_name $exp_name --test_data texture

python predict.py --data /data1/hyunwook/data/ood-cv/weather/ \
                  --batch $batch \
                  --ckpt $ckpt \
                  --exp_name $exp_name --test_data weather

zip -j $out_path results/$exp_name/*.csv
mv $out_path results/$exp_name/
