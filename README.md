# VS-CNN

This is the code for the paper:

**[Visual Sentiment Analysis for Review Images with Item-Oriented and User-Oriented CNN](https://www.researchgate.net/publication/320541140_Visual_Sentiment_Analysis_for_Review_Images_with_Item-Oriented_and_User-Oriented_CNN)**
<br>
[Quoc-Tuan Truong](http://www.qttruong.info/) and [Hady W. Lauw](http://www.hadylauw.com/)
<br>
Presented at [MM 2017](http://www.acmmm.org/2017/)

## Hardware

- CPU Intel(R) Xeon(R) E5-2650 v4 @ 2.20GHz
- DRAM 256 GiB
- GPU NVIDIA Tesla P100

## Requirements

- Python 3
- Tensorflow >= 1.12
- Scikit-learn >= 0.20
- Tqdm >= 4.28

Install dependencies:

```bash
pip3 install -r requirements.txt
```
**Note:**
Tensorflow GPU is installed by default. If you do not have GPU on your machine, please [install](https://www.tensorflow.org/install) CPU version instead.

Download data and pre-trained weights:

```bash
chmod +x download.sh
./download.sh
```

## Experiments

- Train and evaluate the base model VS-CNN:

```bash
python3 train_base.py --dataset [user,business]
```

- Train and evaluate the factor models, iVS-CNN (business) and uVS-CNN (user):

```bash
python3 train_factor.py --dataset [user,business] --factor_layer [conv1,conv3,conv5,fc7] --num_factors 16
```

**Note:**
The factor models use trained weights of the base models for initialization. If you have not trained the base models, pre-trained weights are provided and need to be extracted before training.

```bash
unzip -qq weights.zip
```

- Train and evaluate Naive Bayes baseline:

```bash
python3 train_nb.py --dataset [user,business]
```

## Contact

Questions and discussion are welcome: www.qttruong.info