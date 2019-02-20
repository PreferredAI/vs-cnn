# VS-CNN

This is the code for the paper:

**[Visual Sentiment Analysis for Review Images with Item-Oriented and User-Oriented CNN](https://www.researchgate.net/publication/320541140_Visual_Sentiment_Analysis_for_Review_Images_with_Item-Oriented_and_User-Oriented_CNN)**
<br>
[Quoc-Tuan Truong](http://www.qttruong.info/) and [Hady W. Lauw](http://www.hadylauw.com/)
<br>
Presented at [MM 2017](http://www.acmmm.org/2017/)

We provide:

- Code to train and evaluate the models
- [Data](https://goo.gl/cBF5rn) used for the experiments
- [Pre-trained weights](https://goo.gl/nxnsUx) of the base models

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
- [Pre-trained weights of AlexNet](https://www.cs.toronto.edu/~guerzhoy/tf_alexnet/bvlc_alexnet.npy) for initialization

## Training and Evaluation

- Base model:

```bash
python train_base.py --dataset [user,business]
```

- Factor model:

To train the factor models, we need pre-trained weights from the base models for initialization. If you want to save time, the weights can be downloaded from [here](https://goo.gl/nxnsUx).

```bash
python train_factor.py --dataset [user,business] --factor_layer [conv1,conv3,conv5,fc7] --num_factors 16
```