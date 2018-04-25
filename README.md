# DRML_caffe_python

Deep Region and Multi-label Learning for Facial Action Unit Detection

RL：Region Learning
ML：Multi-label Learning

## Datasets
DISFA   

ck++     

BP4D(need professor to apply for)

## Steps
preprocess data: use opencv to capture frames

step01 write_solver.py: write solver.prototxt

step02 write_prototxt.py: write prototxt of train, test and deploy

step03 main.py train : train model

step04 test_caffemodel.py: test model

## Note
This paper created four new layers:

MultilabelDataLayer,   BoxLayer,   SpliceLayer,   MultilabelSigmoidCrossEntropyLossLayer

I trained on disfa dataset, but the result is not good, I think there is something wrong with maybe the forward and backward of MultilabelSigmoidCrossEntropyLossLayer.

I have used sigmoid loss to train, setting the label as 0 and 1, but considering this is a multi-label task, using 0 and 1 is difficult to test the task for the sigmoid output is larger than 0. So I changed the loss layer. 

If you are interested in this paper, welcome to contact me to discuss.

