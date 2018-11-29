# VS-CNN

This is the code for the paper:

**[Visual Sentiment Analysis for Review Images with Item-Oriented and User-Oriented CNN](https://www.researchgate.net/publication/320541140_Visual_Sentiment_Analysis_for_Review_Images_with_Item-Oriented_and_User-Oriented_CNN)**,
<br>
[Quoc-Tuan Truong](http://www.qttruong.info/) and [Hady W. Lauw](http://www.hadylauw.com/),
<br>
Presented at [MM 2017](http://www.acmmm.org/2017/)

We provide:

- Code to [train](#training) and [evaluate](#evaluation) the models
- [Data](https://goo.gl/cBF5rn) used for the experiments

If you find the code and data useful in your research, please cite:

```
@inproceedings{vs-cnn,
  title={Visual sentiment analysis for review images with item-oriented and user-oriented CNN},
  author={Truong, Quoc-Tuan and Lauw, Hady W},
  booktitle={Proceedings of the ACM on Multimedia Conference},
  year={2017},
}
```

## Requirements

- Python 3
- Tensorflow > 1.0
- Tqdm
- [Pre-trained weights of AlexNet](https://www.cs.toronto.edu/~guerzhoy/tf_alexnet/) for initialization

## Training

Train the base model:

```bash
python train_base.py --dataset [user,business] --num_epochs 20 --batch_size 64 --learning_rate 0.0001 --lambda_reg 0.0005
```

Train the factor model:

```bash
python train_factor.py --dataset [user,business] --num_factors 16 --num_epochs 20 --learning_rate 0.0001 --lambda_reg 0.0005
```


## Evaluation

Evaluate the base model:

```bash
python eval_base.py --dataset [user,business] --batch_size 64
```

Evaluate the factor model:

```bash
python eval_factor.py --dataset [user,business] --num_factors 16
```
