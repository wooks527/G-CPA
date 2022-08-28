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

### Training and Validation

```
python trainer.py --data $DATA_DIR \
                  --batch 128 --epoch 30 \
                  --name resnet50-e50
```

### Prediction

#### Prediction Only
```
python predict.py --data $DATA_DIR \
                  --batch 128 \
                  --ckpt $WEIGHT_PATH \
                  --exp_name resnet50-e50 \
                  --test_data iid
```

#### Prediction with Submission `.csv` File

```
bash predict.sh 256 \
                $WEIGHT_PATH \
                resnet50-e50 \
                first_submit.zip
```

## Contributors

- Hyunwook Kim
- ...
