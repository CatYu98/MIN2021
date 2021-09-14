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

        pretrained_emb.close()

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
        # aspect_cnn.add(Capsule(args.num_capsule, 300, num_routing=3, share_weights=True, activation='squash'))
    aspect_dense = Dense(nb_class, activation='softmax', name='aspect_dense')

    # OTE specific layers
    opinion_cnn = Sequential()
    for a in range(args.aspect_layers):
        opinion_cnn.add(Dropout(args.dropout_prob))
        opinion_cnn.add(Conv1DWithMasking(filters=args.cnn_dim, kernel_size=5, \
                  activation='relu', padding='same', kernel_initializer=my_init, name='opinion_cnn_%s'%a))
        # opinion_cnn.add(Capsule(args.num_capsule, 300, num_routing=3, share_weights=True, activation='squash'))
    opinion_dense = Dense(nb_class, activation='softmax', name='opinion_dense')


    # ASC specific layers
    sentiment_cnn = Sequential()
    for b in range(args.senti_layers):
        sentiment_cnn.add(Dropout(args.dropout_prob))
        sentiment_cnn.add(Conv1DWithMasking(filters=args.cnn_dim, kernel_size=5, \
                  activation='relu', padding='same', kernel_initializer=my_init, name='sentiment_cnn_%s'%b))
        # sentiment_cnn.add(Capsule(args.num_capsule, 300, num_routing=3, share_weights=True, activation='squash'))

    sentiment_att = Self_attention(0, name='sentiment_att')
    sentiment_dense = Dense(3, activation='softmax', name='sentiment_dense')


    # re-encoding layer
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
        opinion_output = sentence_output
        sentiment_output = sentence_output

        ### ATE ###
        if args.aspect_layers > 0:
            aspect_output = aspect_cnn(aspect_output)
        aspect_embedding = aspect_output

        ### OTE ###
        if args.opinion_layers > 0:
            opinion_output = opinion_cnn(opinion_output)
        opinion_embedding = opinion_output
        
        ### pair-attention
        aspect2opinion = Lambda(lambda x : tf.matmul(tf.nn.l2_normalize(x[0], -1), tf.nn.l2_normalize(x[1], -1), adjoint_b=True))([aspect_embedding, opinion_embedding])
        aspect_att_opinion = Lambda(lambda x : softmask_2d(x, sentence_mask))(aspect2opinion)
        aspect2opinion_embedding = Lambda(lambda x: tf.concat([x[0], tf.matmul(x[1], x[2])], -1))([aspect_embedding, aspect_att_opinion, opinion_embedding])

        opinion2aspect = Lambda(lambda x : tf.matmul(tf.nn.l2_normalize(x[0], -1), tf.nn.l2_normalize(x[1], -1), adjoint_b=True))([opinion_embedding, aspect_embedding])
        opinion_att_aspect = Lambda(lambda x : softmask_2d(x, sentence_mask))(opinion2aspect)
        opinion2aspect_embedding = Lambda(lambda x: tf.concat([x[0], tf.matmul(x[1], x[2])], -1))([opinion_embedding, opinion_att_aspect, aspect_embedding])
        
        ### Decode ###
        aspect_output = Concatenate()([word_embeddings, aspect_output, aspect2opinion_embedding])
        aspect_output = Concatenate()([word_embeddings, aspect_output])
        aspect_output = Dropout(args.dropout_prob)(aspect_output)

        opinion_output = Concatenate()([word_embeddings, opinion_output, opinion2aspect_embedding])
        opinion_output = Concatenate()([word_embeddings, opinion_output])
        opinion_output = Dropout(args.dropout_prob)(opinion_output)

        aspect_labels = aspect_dense(aspect_output)
        opinion_labels = opinion_dense(opinion_output)

        aspect_labels0 = Lambda(lambda x: tf.expand_dims(x, -1))(aspect_labels)
        opinion_labels0 = Lambda(lambda x: tf.expand_dims(x, -1))(opinion_labels)

        aspect_list.append(aspect_labels0)
        opinion_list.append(opinion_labels0)

        ### Consistent weighting ###
        pred_ae_pos = Lambda(lambda x: x[:, :, 1]+x[:, :, 2])(aspect_labels) # None, 75
        gold_ae_pos = ae_mask
        pred_ae_weight = Lambda(lambda ex: tf.map_fn(fn=lambda x:Weight(x, num=3, a=3, len = args.num_capsule),elems= ex,dtype=tf.float32))(pred_ae_pos)
        gold_ae_weight = Lambda(lambda ex: tf.map_fn(fn=lambda x:Weight(x, num=3, a=3, len = args.num_capsule),elems= ex,dtype=tf.float32))(gold_ae_pos)
        ae_weight = Add()([Multiply()([p_gold_op, gold_ae_weight]), Multiply()([Subtract()([ones_tensor,p_gold_op]),pred_ae_weight])])
        ae_embedding = Multiply()([init_shared_features, ae_weight])

        pred_oe_pos = Lambda(lambda x: x[:, :, 1]+x[:, :, 2])(opinion_labels) # None, 75
        gold_oe_pos = oe_mask
        pred_oe_weight = Lambda(lambda ex: tf.map_fn(fn=lambda x:Weight(x, num=3, a=3, len = args.num_capsule),elems= ex,dtype=tf.float32))(pred_oe_pos)
        gold_oe_weight = Lambda(lambda ex: tf.map_fn(fn=lambda x:Weight(x, num=3, a=3, len = args.num_capsule),elems= ex,dtype=tf.float32))(gold_oe_pos)
        oe_weight = Add()([Multiply()([p_gold_op, gold_oe_weight]), Multiply()([Subtract()([ones_tensor,p_gold_op]),pred_oe_weight])])
        oe_embedding = Multiply()([init_shared_features, oe_weight])

        ### ASC ###
        if args.senti_layers > 0:
            sentiment_output = sentiment_cnn(sentiment_output)
        sentiment_output = Concatenate()([sentiment_output, ae_embedding])
        sentiment_output = sentiment_att([sentiment_output, opinion_labels])
        sentiment_output = Concatenate()([init_shared_features, sentiment_output, oe_embedding])

        sentiment_output = Dropout(args.dropout_prob)(sentiment_output)
        sentiment_labels = sentiment_dense(sentiment_output)
        sentiment_probs0 = Lambda(lambda x: tf.expand_dims(x, -1))(sentiment_labels)
        sentiment_list.append(sentiment_probs0)


        ### Info Feedback ###
        sentence_output = Concatenate()([sentence_output, aspect_output, opinion_output, sentiment_output])

        sentence_output = enc(sentence_output)

    
    if args.interactions == 2:
        aspect_labels = Concatenate()([aspect_list[0],aspect_list[1],aspect_list[2]])
        opinion_labels = Concatenate()([opinion_list[0],opinion_list[1],opinion_list[2]])
        sentiment_labels = Concatenate()([sentiment_list[0],sentiment_list[1],sentiment_list[2]])
    elif args.interactions == 3:
        aspect_labels = Concatenate()([aspect_list[0],aspect_list[1],aspect_list[2],aspect_list[3]])
        opinion_labels = Concatenate()([opinion_list[0],opinion_list[1],opinion_list[2],opinion_list[3]])
        sentiment_labels = Concatenate()([sentiment_list[0],sentiment_list[1],sentiment_list[2],sentiment_list[3]])
    elif args.interactions == 1:
        aspect_labels = Concatenate()([aspect_list[0],aspect_list[1]])
        opinion_labels = Concatenate()([opinion_list[0],opinion_list[1]])
        sentiment_labels = Concatenate()([sentiment_list[0],sentiment_list[1]])
    elif args.interactions == 4:
        aspect_labels = Concatenate()([aspect_list[0],aspect_list[1],aspect_list[2],aspect_list[3], aspect_list[4]])
        opinion_labels = Concatenate()([opinion_list[0],opinion_list[1],opinion_list[2],opinion_list[3],opinion_list[4]])
        sentiment_labels = Concatenate()([sentiment_list[0],sentiment_list[1],sentiment_list[2],sentiment_list[3],sentiment_list[4]])
    elif args.interactions == 5:
        aspect_labels = Concatenate()([aspect_list[0],aspect_list[1],aspect_list[2],aspect_list[3], aspect_list[4], aspect_list[5]])
        opinion_labels = Concatenate()([opinion_list[0],opinion_list[1],opinion_list[2],opinion_list[3],opinion_list[4], aspect_list[5]])
        sentiment_labels = Concatenate()([sentiment_list[0],sentiment_list[1],sentiment_list[2],sentiment_list[3],sentiment_list[4], aspect_list[5]])
    elif args.interactions == 0:
        aspect_labels = Concatenate()([aspect_list[0],aspect_list[0]])
        opinion_labels = Concatenate()([opinion_list[0],opinion_list[0]])
        sentiment_labels = Concatenate()([sentiment_list[0],sentiment_list[0]])

    aspect_labels = Lambda(lambda x: tf.reduce_mean(x, -1))(aspect_labels)
    opinion_labels = Lambda(lambda x: tf.reduce_mean(x, -1))(opinion_labels)
    sentiment_labels = Lambda(lambda x: tf.reduce_mean(x, -1))(sentiment_labels)

    aspect_model = Model(inputs=[sentence_input, sentence_mask, ae_mask, oe_mask, as_mask, p_gold_op_, ones_tensor_], outputs=[aspect_labels, opinion_labels, sentiment_labels])


    logger.info('Initializing lookup table')
    
    emb_path_gen = '../glove/glove.840B.300d.txt'
    emb_path_domain = '../domain_specific_emb/%s.txt'%(args.domain)



    aspect_model.get_layer('word_emb').set_weights(init_emb(aspect_model.get_layer('word_emb').get_weights(), vocab, emb_path_gen, emb_path_domain))

    logger.info('  Done')
    
    return aspect_model



