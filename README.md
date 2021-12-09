
## Title
Making Flexible Use of Subtasks: A Multiplex Interaction Network for Unified Aspect-based Sentiment Analysis
Accepted by Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021.
Uploaded by Guoxin Yu .

## Training and evaluation
Excute the command below for training and evaluating MIN.
```
CUDA_VISIBLE_DEVICES="0" python train.py 
```

## Model
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
Please download the glove.840B.300d.txt in glove.

## Cite
Thanks for code from:
```
@inproceedings{yu2021making,
  title={Making Flexible Use of Subtasks: A Multiplex Interaction Network for Unified Aspect-based Sentiment Analysis},
  author={Yu, Guoxin and Ao, Xiang and Luo, Ling and Yang, Min and Sun, Xiaofei and Li, Jiwei and He, Qing},
  booktitle={Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021},
  pages={2695--2705},
  year={2021}
} 
```
