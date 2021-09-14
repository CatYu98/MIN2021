import keras.backend as K
from keras.engine.topology import Layer
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.layers.convolutional import Conv1D
import tensorflow as tf
import numpy as np
from keras.layers import Lambda, Multiply, Subtract, Add

class Self_attention(Layer):
    def __init__(self, use_opinion,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        """
        Keras Layer that implements an Content Attention mechanism.
        Supports Masking.
        """
       
        self.use_opinion = use_opinion
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(Self_attention, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[0][-1]
        self.steps = input_shape[0][-2]
     
        self.W = self.add_weight(shape = (input_dim, input_dim),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight(shape = (input_dim,),
                                     initializer='zeros',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, x, mask):
        return mask

    def call(self, input_tensor, mask):
        x = input_tensor[0]
        #gold_opinion = input_tensor[1]
        predict_opinion = input_tensor[1]
        #gold_prob = input_tensor[3]
        mask = mask[0]
        assert mask is not None

        x_tran = K.dot(x, self.W)
        if self.bias:
            x_tran += self.b 

        x_transpose = K.permute_dimensions(x, (0,2,1))
        weights = K.batch_dot(x_tran, x_transpose)

     
        location = np.abs(np.tile(np.array(range(self.steps)), (self.steps,1)) - np.array(range(self.steps)).reshape(self.steps,1))
        loc_weights = 1.0 / (location+K.epsilon())
        loc_weights *= K.cast((location!=0), K.floatx())
        weights *= loc_weights

        # gold_opinion_ = gold_opinion[:,:,1]+gold_opinion[:,:,2]
        # if self.use_opinion==1:
        #     predict_opinion_ = predict_opinion[:,:,1]+predict_opinion[:,:,2]
        #     opinion_weights = predict_opinion_
        #     opinion_weights = K.expand_dims(opinion_weights, axis=-2)
        #     weights *= opinion_weights

        weights = K.tanh(weights)
        weights = K.exp(weights)
        weights *= (np.eye(self.steps)==0)

        if mask is not None:
            mask  = K.expand_dims(mask, axis=-2)
            mask = K.repeat_elements(mask, self.steps, axis=1)
            weights *= K.cast(mask, K.floatx())

        weights /= K.cast(K.sum(weights, axis=-1, keepdims=True) + K.epsilon(), K.floatx())

        output = K.batch_dot(weights, x)
        return output

class Pair_attention(Layer):
    def __init__(self, use_opinion,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        """
        Keras Layer that implements an Content Attention mechanism.
        Supports Masking.
        """
       
        self.use_opinion = use_opinion
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(Pair_attention, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[0][-1]
        self.steps = input_shape[0][-2]
     
        self.W = self.add_weight((input_dim, input_dim),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_dim,),
                                     initializer='zeros',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, x, mask):
        return mask

    def call(self, input_tensor, mask):
        x = input_tensor[0]
        y = input_tensor[1]
        mask = mask[0]
        mask = input_tensor[2]
        assert mask is not None

        x_tran = K.dot(x, self.W)
        if self.bias:
            x_tran += self.b 

        x_transpose = K.permute_dimensions(y, (0,2,1))
        weights = K.batch_dot(x_tran, x_transpose)
     
        '''location = np.abs(np.tile(np.array(range(self.steps)), (self.steps,1)) - np.array(range(self.steps)).reshape(self.steps,1))
        loc_weights = 1.0 / (location+K.epsilon())
        loc_weights *= K.cast((location!=0), K.floatx())
        weights *= loc_weights'''

        weights = K.tanh(weights)
        weights = K.exp(weights)
        weights *= (np.eye(self.steps)==0)

        '''if mask is not None:
            mask  = K.expand_dims(mask, axis=-2)
            mask = K.repeat_elements(mask, self.steps, axis=1)
            weights *= K.cast(mask, K.floatx())'''

        weights /= K.cast(K.sum(weights, axis=-1, keepdims=True) + K.epsilon(), K.floatx())

        output = K.batch_dot(weights, y)
        return output
  
class Conv1DWithMasking(Conv1D):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(Conv1DWithMasking, self).__init__(**kwargs)
    
    def compute_mask(self, x, mask):
        return mask

def squash(x, axis=-1):
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    scale = K.sqrt(s_squared_norm)/ (0.5 + s_squared_norm)
    return scale * x

def softmax(x, axis=-1):
    ex = K.exp(x - K.max(x, axis=axis, keepdims=True))
    return ex/K.sum(ex, axis=axis, keepdims=True)

class Capsule_SR(Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, share_weights=False, activation='squash', **kwargs):
        super(Capsule_SR, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.num_routing = routings
        self.share_weights = share_weights
        self.supports_masking = True
        if activation == 'squash':
            self.activation = squash
        else:
            self.activation = activations.get(activation)

    def build(self, input_shape):
        super(Capsule_SR, self).build(input_shape)
        input_dim_capsule = input_shape[-1]
        input_num_capsule = input_shape[-2]

        self.W_pose = self.add_weight(name='capsule_kernel',
                                    shape=(1, input_dim_capsule,
                                        self.num_capsule * self.dim_capsule),
                                    initializer='glorot_uniform',
                                    trainable=True)

        self.W_route = self.add_weight(name='route_kernel',
                                    shape=(1,input_dim_capsule, self.num_capsule),
                                    initializer='glorot_uniform',
                                    trainable=True)
    def call(self, u_vecs):
        #logger.info("use sr capsule")
        u_hat_vecs = K.conv1d(u_vecs, self.W_pose)
        #print("u_hat_vecs",u_hat_vecs.shape)
        batch_size = K.shape(u_vecs)[0]
        input_num_capsule = K.shape(u_vecs)[1]
        input_dim_capsule = K.shape(u_vecs)[2]
        u_hat_vecs = K.reshape(u_hat_vecs, (batch_size, input_num_capsule,
                                            self.num_capsule, self.dim_capsule))
        u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))
        #final u_hat_vecs.shape = [None, num_capsule, input_num_capsule, dim_capsule]
        #u_vec [None, input_num_capsule, input_dim_capsule]
        b = K.conv1d(u_vecs, self.W_route,strides=1,data_format=None) #shape = [None, num_capsule, input_num_capsule]
        b=K.reshape(b,(K.shape(b)[0], K.shape(b)[2],K.shape(b)[1]))
        c = K.sum(b,axis=1)

        c = softmax(b, 1)
        o = K.batch_dot(c, u_hat_vecs, [1, 1])
        if K.backend() == 'theano':
            o = K.sum(o, axis=1)

        return self.activation(o)

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)

def Position():
    def func(x):
        ii, jj, tt = x.shape()
        weight = x
        for i in range(ii):
            aspect = []
            for j in range(jj):
                for t in range(tt):
                    if x[i][j][t] == 1:
                        aspect.append(t)
            #for a in aspect:
                for b in range(t):
                    relative_pos = t - b
                    #weight.append(1+max(0, 3+text_len//10 - relative_pos ))
                    weight[i][j][b] = 1+max(0, 3+text_len//10 - relative_pos )
                for b in range(t, jj):
                    relative_pos = b - t
                    #weight.append(1+max(0, 3+text_len//10 - relative_pos))
                    weight[i][j][b] = 1+max(0, 3+text_len//10 - relative_pos)
        return weight
        return layers.Lambda(func)

def Weight(x, num = 3, a = 3, len = 78):
    #x is (78, 1)
    
    ### First   
    '''idx = Lambda(lambda x: tf.argmax(x,0))(x) # the index of the first aspect (1,)
    idx = Lambda(lambda x: tf.expand_dims(x, -1))(idx) # (1, 1)
    idx = Lambda(lambda x: tf.tile(x, [78,1]))(idx) #(78, 1)
    #indices = Lambda(lambda x: tf.nn.top_k(x, 3).indices)(idx) # (3,) Slice
    w1 = Lambda(lambda x: tf.ones_like(x))(x)
    w2 = np.arange(0,78)
    w2 = tf.convert_to_tensor(w2)
    w2 = Lambda(lambda x: tf.expand_dims(x, -1))(w2)
    Cas = Lambda(lambda x: tf.cast(x, tf.float32))
    a = Cas(w1)
    b = Cas(idx)
    c = Cas(w2)
    w3 = Multiply()([a, b])
    w3 = Subtract()([w3,c])
    w3 = Lambda(lambda x: tf.abs(x))(w3)'''
    Cas = Lambda(lambda x: tf.cast(x, tf.float32))
    ### get aspect_index ###
    index = Lambda(lambda x: tf.nn.top_k(x, num).indices)(x) # get aspect index, (num,)
    index = Lambda(lambda x: tf.expand_dims(x, -1))(index) # (num, )
    aspect_index = Lambda(lambda x: tf.tile(x, [1,len]))(index) # (num, 78)
    aspect_index = Lambda(lambda x: tf.transpose(x))(aspect_index) # (78, num)

    ### get pro for aspect_index
    '''aspect_pro = index[0]
    for i in range(num):
        if i!= 0:
            aspect_pro = Lambda(lambda x: tf.concat(x, 0))([aspect_pro, index[i]])
    aspect_pro = Lambda(lambda x: tf.expand_dims(x, -1))(aspect_pro)
    aspect_pro = Lambda(lambda x: tf.tile(x, [1,78]))(aspect_pro)
    aspect_pro = Lambda(lambda x: tf.transpose(x))(aspect_pro)'''


    ### get sentence_index for subtract ###
    sentence_index = np.arange(0,len)
    sentence_index = tf.convert_to_tensor(sentence_index) #(78,)
    sentence_index = Lambda(lambda x: tf.expand_dims(x, -1))(sentence_index) # (78, 1)
    sentence_index = Lambda(lambda x: tf.tile(x, [1, num]))(sentence_index) # (78, num)

    #aspect_pro = Cas(aspect_pro)
    aspect_index = Cas(aspect_index)
    sentence_index = Cas(sentence_index)
    w1 = Subtract()([aspect_index, sentence_index]) 
    w1 = Lambda(lambda x: tf.abs(x))(w1) # distance (78, num)

    w1 = Lambda(lambda x: tf.reduce_sum(x, 1))(w1) #  sum of distance for num (78,)
    w1 = Lambda(lambda x: tf.expand_dims(x, -1))(w1) # (78, 1)
    w4 = np.ones([len,1], np.int)
    w4 = tf.convert_to_tensor(w4*(-1))
    w4 = Cas(w4)
    w4 = Multiply()([w1, w4]) # - distance 

    w2 = np.ones([len,1], np.int) # 
    w2 = tf.convert_to_tensor(w2*( a * num )) # A 
    w2 = Cas(w2)

    w2 = Add()([w2, w4]) # A - Sum of dis

    weight = Lambda(lambda x: tf.nn.l2_normalize(x, 0))(w2) # normalize 
    return weight
class Weight_layer(Layer):
    def __init__(self, len = 78 , num = 3, **kwargs):
        super(Weight_layer, self).__init__(**kwargs)
        self.len = len
        self.supports_masking = True
        self.num = num

    def build(self, input_shape):
        super(Weight_layer, self).build(input_shape)
        input_dim = 1
        input_num = input_shape[-1]

        self.w2 = self.add_weight(name='kernel',
                            shape=(1,1,1),
                            initializer='glorot_uniform',
                            trainable=True)

        #self.num = self.add_weight(name = 'top_k', shape = (1), initializer='glorot_uniform', trainable=True)
        #self.w2 = self.add_weight(name = 'a', shape = input_shape, initializer='zeros', trainable=True)
        #self.num = self.add_weight(name = 'top_k', shape = (input_dim, input_dim), initializer='glorot_uniform', trainable=True)

    def call(self, x):

        #self.w2 = tf.cast(self.w2, tf.float32)
        
        # x (None, len)
        index = tf.nn.top_k(x, self.num) 
        index = tf.expand_dims(index[0], -1) # (None, num, 1)
        aspect_index = tf.tile(index, [1,1,self.len]) # (None, num, len)
        aspect_index = tf.transpose(aspect_index, [0,2,1]) # (None, len, num)
        aspect_index = tf.cast(aspect_index, tf.float32)
       
        sentence_index = tf.where(tf.equal(aspect_index, aspect_index), aspect_index, aspect_index)

        w1 = aspect_index - sentence_index
        w1 = tf.abs(w1)
        w1 = tf.reduce_sum(w1, 2)
        w1 = tf.expand_dims(w1, -1)

        w4 = tf.ones_like(w1)
        w4 = tf.cast(w4, tf.float32)
        w4 = w1 * w4

        w2 = K.conv1d(w4, self.w2, strides=1, data_format=None)

        w3 = w2 + w4 # A - Sum of dis

        weight = tf.nn.l2_normalize(w3, -1) # normalize
        return weight

    def compute_output_shape(self, input_shape):
        return (None,self.len, 1)


def softmask_2d(x, mask, scale=False):
    if scale == True:
        dim = tf.shape(x)[-1]
        max_x = tf.reduce_max(x, axis=-1, keepdims=True)
        max_x = tf.tile(max_x, [1, 1, dim])
        x -= max_x
    
    # x is [None, 80, 80], mask [None, 80], 80 is sentence length
    length = tf.shape(mask)[1] #length is sentence length
    mask_d1 = tf.tile(tf.expand_dims(mask, 1), [1, length, 1]) # None, None, 80
    y = tf.multiply(tf.exp(x), mask_d1) # None, 80, 80, mask to get attention for words
    sumx = tf.reduce_sum(y, axis=-1, keepdims=True) # None, 80, 1, 
    att = y / (sumx + 1e-10) # None, 80, 80, normalize

    mask_d2 = tf.tile(tf.expand_dims(mask, 2), [1, 1, length]) # None, 80, 80, mask to get attention
    att *= mask_d2 
    return att

class Capsule(Layer):
    def __init__(self, num_capsule, dim_capsule, num_routing=3, share_weights=True, activation='squash', **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.num_routing = num_routing
        self.share_weights = share_weights
        self.supports_masking = True
        if activation == 'squash':
            self.activation = squash
        else:
            self.activation = activations.get(activation)

    def build(self, input_shape):
        super(Capsule, self).build(input_shape)
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(1, input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(input_num_capsule,
                                            input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)

    def call(self, u_vecs):
        if self.share_weights:
            u_hat_vecs = K.conv1d(u_vecs, self.W)
        else:
            u_hat_vecs = K.local_conv1d(u_vecs, self.W, [1], [1])

        batch_size = K.shape(u_vecs)[0]
        input_num_capsule = K.shape(u_vecs)[1]
        u_hat_vecs = K.reshape(u_hat_vecs, (batch_size, input_num_capsule,
                                            self.num_capsule, self.dim_capsule))
        u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))
        

        b = K.zeros_like(u_hat_vecs[:,:,:,0])
        for i in range(self.num_routing):
            c = softmax(b, 1)
            o = K.batch_dot(c, u_hat_vecs, [2, 2])
            if K.backend() == 'theano':
                o = K.sum(o, axis=1)
            if i < self.num_routing - 1:
                o = K.l2_normalize(o, -1)
                b = K.batch_dot(o, u_hat_vecs, [2, 3])
                if K.backend() == 'theano':
                    b = K.sum(b, axis=1)

        return self.activation(o)

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)


class Capsule_SR(Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, share_weights=False, activation='squash', **kwargs):
        super(Capsule_SR, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.num_routing = routings
        self.share_weights = share_weights
        self.supports_masking = True
        if activation == 'squash':
            self.activation = squash
        else:
            self.activation = activations.get(activation)

    def build(self, input_shape):
        super(Capsule_SR, self).build(input_shape)
        input_dim_capsule = input_shape[-1]
        input_num_capsule = input_shape[-2]

        self.W_pose = self.add_weight(name='capsule_kernel',
                                    shape=(1, input_dim_capsule,
                                        self.num_capsule * self.dim_capsule),
                                    initializer='glorot_uniform',
                                    trainable=True)

        self.W_route = self.add_weight(name='route_kernel',
                                    shape=(1,input_dim_capsule, self.num_capsule),
                                    initializer='glorot_uniform',
                                    trainable=True)
    def call(self, u_vecs):
        #logger.info("use sr capsule")
        u_hat_vecs = K.conv1d(u_vecs, self.W_pose)
        #print("u_hat_vecs",u_hat_vecs.shape)
        batch_size = K.shape(u_vecs)[0]
        input_num_capsule = K.shape(u_vecs)[1]
        input_dim_capsule = K.shape(u_vecs)[2]
        u_hat_vecs = K.reshape(u_hat_vecs, (batch_size, input_num_capsule,
                                            self.num_capsule, self.dim_capsule))
        u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))
        #final u_hat_vecs.shape = [None, num_capsule, input_num_capsule, dim_capsule]
        #u_vec [None, input_num_capsule, input_dim_capsule]
        b = K.conv1d(u_vecs, self.W_route,strides=1,data_format=None) #shape = [None, num_capsule, input_num_capsule]
        b=K.reshape(b,(K.shape(b)[0], K.shape(b)[2],K.shape(b)[1]))
        c = K.sum(b,axis=1)

        c = softmax(b, 1)
        o = K.batch_dot(c, u_hat_vecs, [1, 1])
        if K.backend() == 'theano':
            o = K.sum(o, axis=1)

        return self.activation(o)

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)
