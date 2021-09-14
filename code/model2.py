import logging
import keras.backend as K
from keras.layers import Dense, Dropout, Activation, Embedding, Input, Concatenate, Lambda, Multiply, Add, Subtract
from keras.models import Model, Sequential
from my_layers import Conv1DWithMasking, Self_attention, Capsule_SR, Position, Weight, Weight_layer, Capsule, softmask_2d
import numpy as np
import tensorflow as tf

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

def my_init(shape, dtype=K.floatx()):
    return 0.01 * np.random.standard_normal(size=shape)

def create_model(args, vocab, nb_class, overall_maxlen):

    def init_emb(emb_matrix, vocab, emb_file_gen, emb_file_domain):

        print ('Loading pretrained general word embeddings and domain word embeddings ...')

        counter_gen = 0.
        pretrained_emb = open(emb_file_gen)
        for line in pretrained_emb:
            tokens = line.split()
            if len(tokens) != 301:
                continue
            word = tokens[0]
            vec = tokens[1:]
            try:
                emb_matrix[0][vocab[word]][:300] = vec
                counter_gen += 1
            except KeyError:
                pass

        counter_domain = 0.
        pretrained_emb = open(emb_file_domain)
        for line in pretrained_emb:
            tokens = line.split()
            if len(tokens) != 101:
                continue
            word = tokens[0]
            vec = tokens[1:]
            try:
                emb_matrix[0][vocab[word]][300:] = vec
                counter_domain += 1
            except KeyError:
                pass

        return emb_matrix

    vocab_size = len(vocab)

###################################
# Inputs 
###################################
    sentence_input = Input(shape=(overall_maxlen,), dtype='int32', name='sentence_input')
    sentence_mask = Input(shape=(overall_maxlen,), dtype=K.floatx(), name='sentence_mask')
    ae_mask = Input(shape=(overall_maxlen,), dtype='int32', name='ae_mask')
    oe_mask = Input(shape=(overall_maxlen,), dtype='int32', name='oe_mask')
    as_mask = Input(shape=(overall_maxlen,), dtype='int32', name='as_mask')
    p_gold_op_ = Input(shape=(overall_maxlen,), dtype=K.floatx(), name='p_gold_op')
    ones_tensor_ = Input(shape=(overall_maxlen,), dtype=K.floatx(), name='ones_tensor')
    mask = Lambda(lambda x: tf.expand_dims(x, -1))(sentence_mask)

#########################################
# Shared word embedding layer
#########################################
    print ('Word embedding layer')
    word_emb = Embedding(vocab_size, args.emb_dim, mask_zero=True, name='word_emb')
    p_gold_op = Lambda(lambda x: tf.expand_dims(x, -1))(p_gold_op_)
    ones_tensor = Lambda(lambda x: tf.expand_dims(x, -1))(ones_tensor_)
    word_embeddings = word_emb(sentence_input) 
    sentence_output = word_embeddings

######################################
# Shared Encoder
######################################

    for i in range(args.shared_layers):
        print ('Shared CNN layer %s'%i)
        sentence_output = Dropout(args.dropout_prob)(sentence_output)

        if i == 0:
            conv_1 = Conv1DWithMasking(filters=int(args.cnn_dim/2), kernel_size=3, \
              activation='relu', padding='same', kernel_initializer=my_init, name='cnn_0_1')
            conv_2 = Conv1DWithMasking(filters=int(args.cnn_dim/2), kernel_size=5, \
              activation='relu', padding='same', kernel_initializer=my_init, name='cnn_0_2')

            sentence_output_1 = conv_1(sentence_output)
            sentence_output_2 = conv_2(sentence_output)
            sentence_output = Concatenate()([sentence_output_1, sentence_output_2])


        else:
            conv = Conv1DWithMasking(filters=args.cnn_dim, kernel_size=5, \
              activation='relu', padding='same', kernel_initializer=my_init, name='cnn_%s'%i)

            sentence_output = conv(sentence_output)


        word_embeddings = Concatenate()([word_embeddings, sentence_output])

    init_shared_features = sentence_output

#######################################
# Private Layers
#######################################

    # ATE specific layers
    aspect_cnn = Sequential()
    for a in range(args.aspect_layers):
        aspect_cnn.add(Dropout(args.dropout_prob))
        aspect_cnn.add(Conv1DWithMasking(filters=args.cnn_dim, kernel_size=5, \
                  activation='relu', padding='same', kernel_initializer=my_init, name='aspect_cnn_%s'%a))
    aspect_dense = Dense(nb_class, activation='softmax', name='aspect_dense')


    # ASC specific layers
    sentiment_cnn = Sequential()
    for b in range(args.senti_layers):
        sentiment_cnn.add(Dropout(args.dropout_prob))
        sentiment_cnn.add(Conv1DWithMasking(filters=args.cnn_dim, kernel_size=5, \
                  activation='relu', padding='same', kernel_initializer=my_init, name='sentiment_cnn_%s'%b))

    sentiment_att = Self_attention(0, name='sentiment_att')
    sentiment_dense = Dense(3, activation='softmax', name='sentiment_dense')


    enc = Dense(args.cnn_dim, activation='relu', name='enc')

    aspect_list = []
    opinion_list = []
    sentiment_list = []
    aspect_inputs = []
    opinion_inputs = []
    sentiment_inputs = []
    aspect_inputs.append(sentence_output)
    opinion_inputs.append(sentence_output)
    sentiment_inputs.append(sentence_output)

####################################################
# MIN
####################################################

    for i in range(args.interactions+1):
        print ('Interaction number ', i)
        aspect_output = sentence_output
        sentiment_output = sentence_output

        ### ATE ###
        if args.aspect_layers > 0:
            aspect_output = aspect_cnn(aspect_output)
        aspect_embedding = aspect_output

        ### Decode ###
        aspect_output = Concatenate()([word_embeddings, aspect_output])
        aspect_output = Dropout(args.dropout_prob)(aspect_output)

        aspect_probs = aspect_dense(aspect_output)

        aspect_probs0 = Lambda(lambda x: tf.expand_dims(x, -1))(aspect_probs)

        aspect_list.append(aspect_probs0)

        ### Weighting ###
        pred_ae_pos = Lambda(lambda x: x[:, :, 1]+x[:, :, 2])(aspect_probs) # None, 75
        gold_ae_pos = ae_mask
        pred_ae_weight = Lambda(lambda ex: tf.map_fn(fn=lambda x:Weight(x, num=3, a=3, len = args.num_capsule),elems= ex,dtype=tf.float32))(pred_ae_pos)
        gold_ae_weight = Lambda(lambda ex: tf.map_fn(fn=lambda x:Weight(x, num=3, a=3, len = args.num_capsule),elems= ex,dtype=tf.float32))(gold_ae_pos)
        ae_weight = Add()([Multiply()([p_gold_op, gold_ae_weight]), Multiply()([Subtract()([ones_tensor,p_gold_op]),pred_ae_weight])])
        ae_embedding = Multiply()([init_shared_features, ae_weight])

        ### ASC ###
        sentiment_output = Concatenate()([sentiment_output, ae_embedding])
        if args.senti_layers > 0:
            sentiment_output = sentiment_cnn(sentiment_output)
        
        sentiment_output = sentiment_att([sentiment_output, aspect_probs])
        sentiment_output = Concatenate()([init_shared_features, sentiment_output])

        sentiment_output = Dropout(args.dropout_prob)(sentiment_output)
        sentiment_probs = sentiment_dense(sentiment_output)
        sentiment_probs0 = Lambda(lambda x: tf.expand_dims(x, -1))(sentiment_probs)
        sentiment_list.append(sentiment_probs0)
        
        sentence_output = Concatenate()([sentence_output, aspect_output, sentiment_output])

        sentence_output = enc(sentence_output)

    
    if args.interactions == 2:
        aspect_probs = Concatenate()([aspect_list[0],aspect_list[1],aspect_list[2]])
        sentiment_probs = Concatenate()([sentiment_list[0],sentiment_list[1],sentiment_list[2]])
    elif args.interactions == 3:
        aspect_probs = Concatenate()([aspect_list[0],aspect_list[1],aspect_list[2],aspect_list[3]])
        sentiment_probs = Concatenate()([sentiment_list[0],sentiment_list[1],sentiment_list[2],sentiment_list[3]])
    elif args.interactions == 1:
        aspect_probs = Concatenate()([aspect_list[0],aspect_list[1]])
        sentiment_probs = Concatenate()([sentiment_list[0],sentiment_list[1]])
    elif args.interactions == 4:
        aspect_probs = Concatenate()([aspect_list[0],aspect_list[1],aspect_list[2],aspect_list[3], aspect_list[4]])
        sentiment_probs = Concatenate()([sentiment_list[0],sentiment_list[1],sentiment_list[2],sentiment_list[3],sentiment_list[4]])
    elif args.interactions == 0:
        aspect_probs = Concatenate()([aspect_list[0],aspect_list[0]])
        sentiment_probs = Concatenate()([sentiment_list[0],sentiment_list[0]])
    elif args.interactions == 5:
        aspect_probs = Concatenate()([aspect_list[0],aspect_list[1],aspect_list[2],aspect_list[3], aspect_list[4], aspect_list[5]])
        sentiment_probs = Concatenate()([sentiment_list[0],sentiment_list[1],sentiment_list[2],sentiment_list[3],sentiment_list[4], aspect_list[5]])

    aspect_probs = Lambda(lambda x: tf.reduce_mean(x, -1))(aspect_probs)
    sentiment_probs = Lambda(lambda x: tf.reduce_mean(x, -1))(sentiment_probs)

    aspect_model = Model(inputs=[sentence_input, sentence_mask, ae_mask, oe_mask, as_mask, p_gold_op_, ones_tensor_], outputs=[aspect_probs, sentiment_probs])


    logger.info('Initializing lookup table')
    
    emb_path_gen = '../glove/glove.840B.300d.txt'
    emb_path_domain = '../domain_specific_emb/%s.txt'%(args.domain)



    aspect_model.get_layer('word_emb').set_weights(init_emb(aspect_model.get_layer('word_emb').get_weights(), vocab, emb_path_gen, emb_path_domain))

    logger.info('  Done')
    
    return aspect_model
