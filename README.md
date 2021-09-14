## Training and evaluation
Excute the command below for training and evaluating MIN.
```
CUDA_VISIBLE_DEVICES="0" python train.py 
```

## model
model3  three sub-tasks:ATE+OTE+ASC  
model2  two sub-tasks: ATE+ASC  
model1  two sub-tasks: ATE+OTE  
model0  two sub-tasks: OTE+ASC  

You can choose different models and combinations of sub-tasks by changing the parameter named 'tasks'.

Note that there are some unused custom-defiened layers in my_layers.py. Please ignore them.

## Dependencies
* Python 3.6
* Keras 2.3.0
* tensorflow 1.15.0
* numpy 1.19.1

## Note
The current code version is not perfect enough for reference only.
Please download the glove.840B.300d.txt in glove.

## Cite
Thanks for code from:
```
@InProceedings{he_acl2019,
  author    = {He, Ruidan  and  Lee, Wee Sun  and  Ng, Hwee Tou  and  Dahlmeier, Daniel},
  title     = {An Interactive Multi-Task Learning Network for End-to-End Aspect-Based Sentiment Analysis},
  booktitle = {Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics},
  publisher = {Association for Computational Linguistics}
}
```
