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

- CUDA 10.0
- CuDNN 7.5
- Python 3.6
- Tensorflow >=1.12, <2.0
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
Training arguments:
```
--data_dir            DATA_DIR
                      Path to data directory (default: 'data')
--dataset             DATASET
                      Name of dataset business/user (default: 'user')
--learning_rate       LEARNING_RATE
                      Learning rate (default: 0.0001)
--num_epochs          NUM_EPOCHS
                      Number of training epochs (default: 50)
--batch_size          BATCH_SIZE (only for the base model VS-CNN)
                      Number of images per batch (default: 50)
--factor_layer        FACTOR_LAYER (only for factor models iVS-CNN/uVS-CNN)
                      Name of the layer to introduce user/item factors conv1/conv3/conv5/fc7 (default: 'fc7')
--num_factors         NUM_FACTORS (only for factor models iVS-CNN/uVS-CNN)
                      Number of neurons/filters for user/item (default: 16)
--lambda_reg          LAMBDA_REG
                      Lambda hype-parameter for L2 regularization (default: 5e-4)
--dropout_keep_prob   DROPOUT_KEEP_PROB
                      Probability of keeping neurons from dropout (default: 0.5)
--num_threads         NUM_THREADS
                      Number of threads for data processing (default: 8)
--num_checkpoints     NUM_CHECKPOINTS
                      Number of checkpoints to store (default: 5)
--display_step        DISPLAY_STEP
                      Number of steps to display log into TensorBoard (default: 10)
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

## Reproducing Tables and Figures

- To aggregate the results and construct the `Tables 1 to 8` as reported, it requires both iV-CNN and uVS-CNN trained with `k = 8` and `k = 16`, also all the output files to be placed inside the same folder (i.e., result_dir). 
The created tables will be saved within the same folder.
```bash
python3 gen_tables.py --dir result_dir
```

- To create `Figure 5, 7 and 9`, we retrieve the top 300 images for each of the positive and negative classes, then manually cluster them into 4 categories:
```bash
python3 case_study_base.py --dataset [user,business] --num_images 300
```

- For "contrarian" items/users in `Figure 6, 8, and 10`, we can retrieve those users/items and then analyze their images:
```bash
python3 case_study_factor.py --dataset [user,business] --num_items 10 --input_dir selected_images --output_dir retrieved_items
```

## Contact

Questions and discussion are welcome: www.qttruong.info
