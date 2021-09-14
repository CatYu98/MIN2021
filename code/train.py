# -*- coding: utf-8 -*
import argparse
import codecs
import logging
import numpy as np
from time import time
import utils as U
import read as dataset
from evaluation import get_metric, convert_to_list
import math


# Parse arguments

parser = argparse.ArgumentParser()

parser.add_argument("--domain", dest="domain", type=str, metavar='<str>', default='lt', help="domain of the corpus {res, lt, res_15}")
parser.add_argument("-v", "--vocab-size", dest="vocab_size", type=int, metavar='<int>', default=20000, help="Vocab size. '0' means no limit (default=20000)")
parser.add_argument("-a", "--algorithm", dest="algorithm", type=str, metavar='<str>', default='adam', help="Optimization algorithm (rmsprop|sgd|adagrad|adadelta|adam|adamax) (default=adam)")
parser.add_argument("-b", "--batch-size", dest="batch_size", type=int, metavar='<int>', default=8, help="Batch size (default=32)")
parser.add_argument("--epochs", dest="epochs", type=int, metavar='<int>', default=100, help="Number of epochs (default=80)")
parser.add_argument("--validation-ratio", dest="validation_ratio", type=float, metavar='<float>', default=0.2, help="The percentage of training data used for validation")
parser.add_argument("-e", "--embdim", dest="emb_dim", type=int, metavar='<int>', default=400, help="Embeddings dimension (default=dim_general_emb + dim_domain_emb = 400)")
parser.add_argument("-c", "--cnndim", dest="cnn_dim", type=int, metavar='<int>', default=300, help="CNN output dimension. '0' means no CNN layer (default=300)")
parser.add_argument("--dropout", dest="dropout_prob", type=float, metavar='<float>', default=0.5, help="The dropout probability. (default=0.5)")
parser.add_argument("--shared-layers", dest="shared_layers", type=int, metavar='<int>', default=2, help="The number of CNN layers in the shared network")
parser.add_argument("--senti-layers", dest="senti_layers", type=int, metavar='<int>', default=1, help="") 
parser.add_argument("--aspect-layers", dest="aspect_layers", type=int, metavar='<int>', default=1, help="") 
parser.add_argument("--opinion-layers", dest="opinion_layers", type=int, metavar='<int>', default=1, help="")
parser.add_argument("--num-capsule", dest="num_capsule", type=int, metavar='<int>', default=82, help="The number of capsules") 
parser.add_argument("--tasks", dest="tasks", type=int, metavar='<int>', default=3, help="1 denotes AE&OE, 2 denotes AE&AS, 3 denotes AE&OE&AS, 0 denotes OE&AS") 
parser.add_argument("--interactions", dest="interactions", type=int, metavar='<int>', default=3)

parser.add_argument("--seed", dest="seed", type=int, metavar='<int>', default=123, help="Random seed (default=123)")
args = parser.parse_args()

U.print_args(args)
 
from numpy.random import seed 
seed(args.seed) 
from tensorflow import set_random_seed 
set_random_seed(args.seed)

if args.tasks == 3:
    logging.basicConfig(
                        filename='../log/AE_OE_AS/'+str(args.domain)+'/out_batch_'+str(args.batch_size)+'_'+str(args.interactions)+'.log',
                        level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
elif args.tasks == 2:
    logging.basicConfig(
                        filename='../log/AE_AS/'+str(args.domain)+'/out_batch_'+str(args.batch_size)+str(args.interactions)+'out.log',
                        level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
elif args.tasks == 1:
    logging.basicConfig(
                        filename='../log/AE_OE/'+str(args.domain)+'/out_batch_'+str(args.batch_size)+'out.log',
                        level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
elif args.tasks == 0:
    logging.basicConfig(
                        filename='../log/OE_AS/'+str(args.domain)+'/out_batch_'+str(args.batch_size)+'out.log',
                        level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

from keras.preprocessing import sequence
from keras.utils.np_utils import to_categorical
import copy

def convert_label(label, nb_class, maxlen):
    label_ = np.zeros((len(label), maxlen, nb_class))
    mask = np.zeros((len(label), maxlen))
    mask_ = np.zeros((len(label), maxlen))

    for i in range(len(label)):
        for j in range(len(label[i])):
            l = label[i][j]
            label_[i][j][l] = 1
            mask[i][j] = 1
            if l == 1 or l == 2:
                mask_[i][j] = 1
    return label_, mask, mask_

def convert_label_sentiment(label, nb_class, maxlen):
    label_ = np.zeros((len(label), maxlen, nb_class))
    mask = np.zeros((len(label), maxlen)) 
    for i in range(len(label)): 
        for j in range(len(label[i])): 
            l = label[i][j]
            # for background word and word with conflict label, set its sentiment label to [0,0,0]
            # such that we don't consider them in the sentiment classification loss
            if l in [1,2,3]: 
                label_[i][j][l-1] = 1 
                mask[i][j] = 1 
            
    return label_, mask

def shuffle(array_list):
    len_ = len(array_list[0])
    for x in array_list:
        assert len(x) == len_
    p = np.random.permutation(len_)
    return [x[p] for x in array_list]

def batch_generator(array_list, batch_size):
    batch_count = 0
    n_batch = int(len(array_list[0]) / batch_size)
    # print('!!!!!!!!!!')
    # print(batch_size)
    # print(n_batch)
    array_list = shuffle(array_list)

    while True:
        if batch_count == n_batch:
            array_list = shuffle(array_list)
            batch_count = 0

        batch_list = [x[batch_count*batch_size: (batch_count+1)*batch_size] for x in array_list]
        batch_count += 1
        yield batch_list

def split_dev(array_list, ratio=0.2):
    validation_size = int(len(array_list[0]) * ratio)
    array_list = shuffle(array_list)
    dev_sets = [x[:validation_size] for x in array_list]
    train_sets = [x[validation_size:] for x in array_list]
    return train_sets, dev_sets


train_x, train_label_target, train_label_opinion, train_label_polarity, \
test_x, test_label_target, test_label_opinion, test_label_polarity,\
    vocab, overall_maxlen = dataset.prepare_data(args.domain, args.vocab_size)
# prepare data

train_y_aspect = copy.deepcopy(train_label_target)
test_y_aspect = copy.deepcopy(test_label_target)

train_x = sequence.pad_sequences(train_x, maxlen=overall_maxlen, padding='post', truncating='post')
test_x = sequence.pad_sequences(test_x, maxlen=overall_maxlen, padding='post', truncating='post')

train_y_aspect, train_y_mask, train_y_aspect_mask = convert_label(train_y_aspect, 3, overall_maxlen)
test_y_aspect, test_y_mask, test_y_aspect_mask = convert_label(test_y_aspect, 3, overall_maxlen)

train_y_sentiment, train_y_sentiment_mask = convert_label_sentiment(train_label_polarity, 3, overall_maxlen)
test_y_sentiment, test_y_sentiment_mask = convert_label_sentiment(test_label_polarity, 3, overall_maxlen)

train_y_opinion, _, train_y_opinion_mask = convert_label(train_label_opinion, 3, overall_maxlen)
test_y_opinion, _, test_y_opinion_mask = convert_label(test_label_opinion, 3, overall_maxlen)

[train_x, train_y_aspect, train_y_sentiment, train_y_opinion, train_y_mask,  train_y_aspect_mask, train_y_sentiment_mask, train_y_opinion_mask], \
[dev_x, dev_y_aspect, dev_y_sentiment, dev_y_opinion, dev_y_mask, dev_y_aspect_mask, dev_y_sentiment_mask, dev_y_opinion_mask] = \
split_dev([train_x, train_y_aspect, train_y_sentiment, train_y_opinion, train_y_mask, train_y_aspect_mask, train_y_sentiment_mask, train_y_opinion_mask], ratio=args.validation_ratio)


# optimizer

from optimizers import get_optimizer
optimizer = get_optimizer(args)


# Building model
if args.tasks == 1:
    from model1 import create_model
elif args.tasks == 2:
    from model2 import create_model
if args.tasks == 3:
    from model3 import create_model
if args.tasks == 0:
    from model0 import create_model

model = create_model(args, vocab, 3, overall_maxlen)
model.summary()

if args.tasks == 3:
    model.compile(optimizer=optimizer, 
            loss=['categorical_crossentropy',  'categorical_crossentropy', 'categorical_crossentropy'],
            loss_weights=[1., 1., 1.])
else:
    model.compile(optimizer=optimizer, 
            loss=['categorical_crossentropy',  'categorical_crossentropy'],
            loss_weights=[1., 1.])


# Training

def get_prob(epoch_count):
    prob = 5/(5+np.exp(epoch_count/5))
    return prob

from tqdm import tqdm

flag_dev, flag_loss = 0, 0
tem_loss = 100
tem_dev = 0

best_dev, best_loss = 0, 0
best_dev_ae = 0
best_dev_oe = 0
best_dev_as = 0
best_loss_ae = 0
best_loss_oe = 0
best_loss_as = 0

best_test_metric = 0

gen_aspect = batch_generator([train_x, train_y_aspect, train_y_sentiment, train_y_opinion, train_y_mask, train_y_aspect_mask, train_y_sentiment_mask, train_y_opinion_mask], batch_size=args.batch_size)
print(len(train_x))
batches_per_epoch_aspect = len(train_x) / args.batch_size

if args.tasks == 3:     
    for ii in range(args.epochs):
        t0 = time()
        loss, loss_aspect, loss_opinion, loss_sentiment = 0., 0., 0., 0.

        gold_prob = get_prob(ii)
        rnd = np.random.uniform()
        if rnd < gold_prob:
            gold_prob = np.ones((args.batch_size, overall_maxlen))
        else:
            gold_prob = np.zeros((args.batch_size, overall_maxlen))
        
        ones_tensor = np.ones((args.batch_size, overall_maxlen))

        for b in tqdm(range(int(batches_per_epoch_aspect))):
            batch_x, batch_y_ae, batch_y_as, batch_y_op, bacth_mask, batch_ae_mask, batch_as_mask, batch_op_mask = gen_aspect.__next__()            
            loss_, loss_aspect_, loss_opinion_, loss_sentiment_ = model.train_on_batch([batch_x, bacth_mask, batch_ae_mask, batch_op_mask, batch_as_mask, gold_prob, ones_tensor], [batch_y_ae, batch_y_op, batch_y_as])
            loss += loss_ / batches_per_epoch_aspect
            loss_aspect += loss_aspect_ / batches_per_epoch_aspect
            loss_opinion += loss_opinion_ / batches_per_epoch_aspect
            loss_sentiment += loss_sentiment_ / batches_per_epoch_aspect

            flag_loss, flag_dev = 0, 0

            if ii>150 and loss < tem_loss:
                tem_loss = loss
                flag_loss = 1

        tr_time = time() - t0

        logger.info('Epoch %d, train: %is' % (ii, tr_time))
        logger.info('Loss %.8f %.8f %.8f %.8f :'%(loss, loss_aspect, loss_opinion, loss_sentiment))
        print (loss, loss_aspect, loss_opinion, loss_sentiment)

        y_pred_aspect, y_pred_opinion, y_pred_sentiment = model.predict([dev_x, dev_y_mask, dev_y_mask, dev_y_mask, dev_y_mask, np.zeros((len(dev_x), overall_maxlen)), np.ones((len(dev_x), overall_maxlen))])
        
        AE_F1, OE_F1, AS_ACC, AS_F1, O_F1 \
            = get_metric(dev_y_aspect, y_pred_aspect, dev_y_opinion, y_pred_opinion, dev_y_sentiment, y_pred_sentiment, dev_y_mask, 0)

        logger.info('Validation results -- [AE_F1]: %.4f, [OE_F1]: %.4f, [AS_f1]: %.4f, [O_F1]: %.4f' 
                            %(AE_F1, OE_F1, AS_F1, O_F1))


        if O_F1 > best_test_metric:
            logger.info('The O_F1 has risen from %.4f to %.4f.'%(best_test_metric, O_F1))

        y_pred_aspect, y_pred_opinion, y_pred_sentiment = model.predict([test_x, test_y_mask, test_y_mask, test_y_mask, test_y_mask, np.zeros((len(test_x), overall_maxlen)), np.ones((len(test_x), overall_maxlen))])
        AE_F1, OE_F1, AS_ACC, AS_F1, O_F1 \
            = get_metric(test_y_aspect, y_pred_aspect, test_y_opinion, y_pred_opinion, test_y_sentiment, y_pred_sentiment, test_y_mask, 0)
        logger.info('Test results -- [AE_F1]: %.4f, [OE_F1]: %.4f, [AS_f1]: %.4f, [O_F1]: %.4f' 
                            %(AE_F1, OE_F1, AS_F1, O_F1))

elif args.tasks == 2:     
    for ii in range(args.epochs):
        t0 = time()
        loss, loss_aspect, loss_opinion, loss_sentiment = 0., 0., 0., 0.

        gold_prob = get_prob(ii)
        rnd = np.random.uniform()
        if rnd < gold_prob:
            gold_prob = np.ones((args.batch_size, overall_maxlen))
        else:
            gold_prob = np.zeros((args.batch_size, overall_maxlen))
        
        ones_tensor = np.ones((args.batch_size, overall_maxlen))

        for b in tqdm(range(int(batches_per_epoch_aspect))):
            batch_x, batch_y_ae, batch_y_as, batch_y_op, bacth_mask, batch_ae_mask, batch_as_mask, batch_op_mask = gen_aspect.__next__()  
            loss_, loss_aspect_, loss_sentiment_ = model.train_on_batch([batch_x, bacth_mask, batch_ae_mask, batch_op_mask, batch_as_mask, gold_prob, ones_tensor], [batch_y_ae, batch_y_as])
            loss += loss_ / batches_per_epoch_aspect
            loss_aspect += loss_aspect_ / batches_per_epoch_aspect
            loss_sentiment += loss_sentiment_ / batches_per_epoch_aspect

            flag_loss, flag_dev = 0, 0

            if ii>150 and loss < tem_loss:
                tem_loss = loss
                flag_loss = 1

        tr_time = time() - t0

        logger.info('Epoch %d, train: %is' % (ii, tr_time))
        logger.info('Loss %.8f %.8f %.8f :'%(loss, loss_aspect, loss_sentiment))
        print (loss, loss_aspect, loss_sentiment)

        y_pred_aspect, y_pred_sentiment = model.predict([dev_x, dev_y_mask, dev_y_mask, dev_y_mask, dev_y_mask, np.zeros((len(dev_x), overall_maxlen)), np.ones((len(dev_x), overall_maxlen))])
        
        AE_F1, OE_F1, AS_ACC, AS_F1, O_F1 \
            = get_metric(dev_y_aspect, y_pred_aspect, dev_y_aspect, y_pred_aspect, dev_y_sentiment, y_pred_sentiment, dev_y_mask, 0)

        logger.info('Validation results -- [AE_F1]: %.4f, [OE_F1]: %.4f, [AS_f1]: %.4f, [O_F1]: %.4f' 
                            %(AE_F1, OE_F1, AS_F1, O_F1))


        if O_F1 > best_test_metric and ii >50:
            logger.info('The O_F1 has risen from %.4f to %.4f.'%(best_test_metric, O_F1))
            best_test_metric = O_F1
            best_ae = AE_F1
            best_oe = OE_F1
            best_as = AS_F1

        y_pred_aspect, y_pred_sentiment = model.predict([test_x, test_y_mask, test_y_mask, test_y_mask, test_y_mask, np.zeros((len(test_x), overall_maxlen)), np.ones((len(test_x), overall_maxlen))])
        AE_F1, OE_F1, AS_ACC, AS_F1, O_F1 \
            = get_metric(test_y_aspect, y_pred_aspect, test_y_aspect, y_pred_aspect, test_y_sentiment, y_pred_sentiment, test_y_mask, 0)
        logger.info('Test results -- [AE_F1]: %.4f, [OE_F1]: %.4f, [AS_f1]: %.4f, [O_F1]: %.4f' 
                            %(AE_F1, OE_F1, AS_F1, O_F1))
        
elif args.tasks == 1:     
    for ii in range(args.epochs):
        t0 = time()
        loss, loss_aspect, loss_opinion, loss_sentiment = 0., 0., 0., 0.

        gold_prob = get_prob(ii)
        rnd = np.random.uniform()
        if rnd < gold_prob:
            gold_prob = np.ones((args.batch_size, overall_maxlen))
        else:
            gold_prob = np.zeros((args.batch_size, overall_maxlen))
        
        ones_tensor = np.ones((args.batch_size, overall_maxlen))

        for b in tqdm(range(int(batches_per_epoch_aspect))):
            batch_x, batch_y_ae, batch_y_as, batch_y_op, bacth_mask, batch_ae_mask, batch_as_mask, batch_op_mask = gen_aspect.__next__()            
            loss_, loss_aspect_, loss_opinion_ = model.train_on_batch([batch_x, bacth_mask, batch_ae_mask, batch_op_mask, batch_as_mask, gold_prob, ones_tensor], [batch_y_ae, batch_y_op])
            loss += loss_ / batches_per_epoch_aspect
            loss_aspect += loss_aspect_ / batches_per_epoch_aspect
            loss_opinion += loss_opinion_ / batches_per_epoch_aspect

            flag_loss, flag_dev = 0, 0

            if ii>150 and loss < tem_loss:
                tem_loss = loss
                flag_loss = 1

        tr_time = time() - t0

        logger.info('Epoch %d, train: %is' % (ii, tr_time))
        logger.info('Loss %.8f %.8f %.8f :'%(loss, loss_aspect, loss_opinion))
        print (loss, loss_aspect, loss_opinion)

        y_pred_aspect, y_pred_opinion = model.predict([dev_x, dev_y_mask, dev_y_mask, dev_y_mask, dev_y_mask, np.zeros((len(dev_x), overall_maxlen)), np.ones((len(dev_x), overall_maxlen))])
        
        AE_F1, OE_F1, AS_ACC, AS_F1, O_F1 \
            = get_metric(dev_y_aspect, y_pred_aspect, dev_y_opinion, y_pred_opinion, dev_y_sentiment, dev_y_sentiment, dev_y_mask, 0)

        logger.info('Validation results -- [AE_F1]: %.4f, [OE_F1]: %.4f, [AS_f1]: %.4f, [O_F1]: %.4f' 
                            %(AE_F1, OE_F1, AS_F1, O_F1))
    

        if AE_F1+OE_F1 > best_test_metric:
            logger.info('The O_F1 has risen from %.4f to %.4f.'%(best_test_metric, O_F1))
            best_test_metric = AE_F1+OE_F1

        y_pred_aspect, y_pred_opinion = model.predict([test_x, test_y_mask, test_y_mask, test_y_mask, test_y_mask, np.zeros((len(test_x), overall_maxlen)), np.ones((len(test_x), overall_maxlen))])
        AE_F1, OE_F1, AS_ACC, AS_F1, O_F1 \
            = get_metric(test_y_aspect, y_pred_aspect, test_y_opinion, y_pred_opinion, test_y_sentiment, test_y_sentiment, test_y_mask, 0)
        logger.info('Test results -- [AE_F1]: %.4f, [OE_F1]: %.4f, [AS_f1]: %.4f, [O_F1]: %.4f' 
                            %(AE_F1, OE_F1, AS_F1, O_F1))
        
elif args.tasks == 0:     
    for ii in range(args.epochs):
        t0 = time()
        loss, loss_aspect, loss_opinion, loss_sentiment = 0., 0., 0., 0.

        gold_prob = get_prob(ii)
        rnd = np.random.uniform()
        if rnd < gold_prob:
            gold_prob = np.ones((args.batch_size, overall_maxlen))
        else:
            gold_prob = np.zeros((args.batch_size, overall_maxlen))
        
        ones_tensor = np.ones((args.batch_size, overall_maxlen))

        for b in tqdm(range(int(batches_per_epoch_aspect))):
            batch_x, batch_y_ae, batch_y_as, batch_y_op, bacth_mask, batch_ae_mask, batch_as_mask, batch_op_mask = gen_aspect.__next__()            
            loss_, loss_opinion_, loss_sentiment_ = model.train_on_batch([batch_x, bacth_mask, batch_ae_mask, batch_op_mask, batch_as_mask, gold_prob, ones_tensor], [batch_y_op,batch_y_as])
            loss += loss_ / batches_per_epoch_aspect
            loss_opinion += loss_opinion_ / batches_per_epoch_aspect
            loss_sentiment += loss_sentiment_ / batches_per_epoch_aspect

            flag_loss, flag_dev = 0, 0

            if ii>150 and loss < tem_loss:
                tem_loss = loss
                flag_loss = 1

        tr_time = time() - t0

        logger.info('Epoch %d, train: %is' % (ii, tr_time))
        logger.info('Loss %.8f %.8f %.8f :'%(loss, loss_aspect, loss_opinion))
        print (loss, loss_opinion, loss_sentiment)

        y_pred_opinion, y_pred_sentiment = model.predict([dev_x, dev_y_mask, dev_y_mask, dev_y_mask, dev_y_mask, np.zeros((len(dev_x), overall_maxlen)), np.ones((len(dev_x), overall_maxlen))])
        
        AE_F1, OE_F1, AS_ACC, AS_F1, O_F1 \
            = get_metric(dev_y_aspect, dev_y_aspect, dev_y_opinion, y_pred_opinion, dev_y_sentiment, y_pred_sentiment, dev_y_mask, 0)

        logger.info('Validation results -- [AE_F1]: %.4f, [OE_F1]: %.4f, [AS_f1]: %.4f, [O_F1]: %.4f' 
                            %(AE_F1, OE_F1, AS_F1, O_F1))


        if AE_F1+OE_F1 > best_test_metric:
            logger.info('The O_F1 has risen from %.4f to %.4f.'%(best_test_metric, O_F1))
            best_test_metric = AE_F1+OE_F1

        y_pred_opinion, y_pred_sentiment = model.predict([test_x, test_y_mask, test_y_mask, test_y_mask, test_y_mask, np.zeros((len(test_x), overall_maxlen)), np.ones((len(test_x), overall_maxlen))])
        AE_F1, OE_F1, AS_ACC, AS_F1, O_F1 \
            = get_metric(test_y_aspect, test_y_aspect, test_y_opinion, y_pred_opinion, test_y_sentiment, y_pred_sentiment, test_y_mask, 0)
        logger.info('Test results -- [AE_F1]: %.4f, [OE_F1]: %.4f, [AS_f1]: %.4f, [O_F1]: %.4f' 
                            %(AE_F1, OE_F1, AS_F1, O_F1))

