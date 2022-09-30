# OOD-CV Classification

In this project, we proposed a better robust classification models to out-of-distribution shifts in the data called OOD-CV. In the dataset, there are 10 object categories (aeroplane, bus, car, train, boat, bicycle, motorplane, chair, dining table, sofa) from the PASCAL VOC 2012 and ImageNet datasets.

## Table of Contents

- [Challenge Description](#challenge-description)
- [Get Started](#get-started)
- [Contributors](#contributors)

## Challenge Description

<img src="https://www.ood-cv.org/images/motivation-6nuisances_small.jpg"></a>

Deep learning models are usually developed and tested under the implicit assumption that the training and test data are drawn independently and identically distributed (IID) from the same distribution. Overlooking out-of-distribution (OOD) images can result in poor performance in unseen or adverse viewing conditions, which is common in real-world scenarios.

This competition aims to tackle typical computer vision tasks (i.e. Multi-class Classification, Object Detection, ...) on OOD images which follows a different distribution than the training images.

If you want to see more details of the challenge, please check the official challenge website ([Link](https://www.ood-cv.org/)).

## Get Started

### Prerequsite

- Create the virtual environment.

    ```
    conda create -n ood-cv-cls python=3.8 -y
    conda activate ood-cv
    ```

- Install packages.

    ```
    conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -y
    pip -r install requirements.txt
    pre-commit autoupdate
    pre-commit install
    ```

- Copy files related SwinTransformer V2 model.

    ```
    cp models/swinv2/* ~/anaconda3/env/ood-cv-cls/lib/python3.8/site-packages/torchvision/models
    ```

- Download datasets.

    ```
    cd ~/data

    # phase-1
    gdown 1bdBmI3_ZwDuIIipHq8xI5ZpYRhsWMQka
    unzip ROBINv1.1-cls-pose.zip
    mv ROBINv1.1-cls-pose ood-cv-cls
    cd ood-cv-cls
    mkdir val
    cp -r nuisances/*/Images val/
    cd ../


    gdown 1xOxlrTjQb4V2uZFrp1LUdJniUI_ut_gB # phase-2
    unzip OOD-CV-phase2.zip -d ood-cv-phase2
    ```

### Training and Validation

```
python trainer.py --data $DATA_DIR \
                  --model swin_v2_b --num_classes $NUM_CLASSES \
                  --batch $BATCH_SIZE --epoch $EPOCH \
                  --warmup_epochs $WARMUP_EPOCHS \
                  --lr $LEARNING_RATE \
                  --weight_decay $WEIGHT_DECAY \
                  --exp_name $EXP_NAME \
                  --seed $SEED
```

### Prediction

#### Prediction Only
```
python predict.py --data $DATA_DIR \
                  --batch $BATCH_SIZE \
                  --ckpt $WEIGHT_PATH \
                  --exp_name $EXP_NAME \
                  --test_data $TEST_TYPE
```

#### Prediction with Submission `.csv` File

##### Phase-1

```
bash predict.sh $DATA_DIR
                $BATCH_SIZE \
                $WEIGHT_PATH \
                $EXP_NAME \
                $OUTZIP_FNAME
```

## Contributors

- Hyunwook Kim
- Sungmin Park
