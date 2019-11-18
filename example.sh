#!/bin/bash

rm -rf examples/reversed_positive_drink

python3 case_study_factor.py --dataset business --num_items 5 --input_dir examples/positive_drink --output_dir examples/reversed_positive_drink
