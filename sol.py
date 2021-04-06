import tensorflow as tf
import math
pi = tf.constant(math.pi)

##############################################################################

acc = lambda TN,FP,FN,TP: tf.math.divide_no_nan(TP+TN,TN+FP+FN+TP)
prec = lambda TN,FP,FN,TP: tf.math.divide_no_nan(TP,FP+TP)
rec = lambda TN,FP,FN,TP: tf.math.divide_no_nan(TP,FN+TP)
spec = lambda TN,FP,FN,TP: tf.math.divide_no_nan(TN,FP+TN)
f1s = lambda TN,FP,FN,TP: 2.*tf.math.divide_no_nan(prec(TN,FP,FN,TP)*\
                          rec(TN,FP,FN,TP),prec(TN,FP,FN,TP)+rec(TN,FP,FN,TP))
tss = lambda TN,FP,FN,TP: rec(TN,FP,FN,TP)+spec(TN,FP,FN,TP)-1.
csi = lambda TN,FP,FN,TP: tf.math.divide_no_nan(TP,FN+FP+TP)
hss1 = lambda TN,FP,FN,TP: tf.math.divide_no_nan(TP-FP,FN+TP)
hss2 = lambda TN,FP,FN,TP: tf.math.divide_no_nan(2.*((TP*TN)-(FP*FN)),((TP+FN)\
                            *(FN+TN))+((TP+FP)*(TN+FP)))
    
F_unif = lambda x: x
F_cos = lambda x,mu,delta: tf.where(x<mu-delta,0.,tf.where(x>mu+delta,1.,\
        0.5*(1.+tf.math.divide(x-mu,delta)+1./pi*tf.math.sin(pi*\
         tf.math.divide(x-mu,delta)))))



def SOL(score = 'accuracy', distribution = 'uniform',\
        mu = 0.5, delta = 0.1, mode = 'average'):

    """
    
    Score-Oriented Loss (SOL)

    Compute the expected confusion matrix defined by the following elements
           n
      TP = ∑  y_i * F(p_i)
           n
      TN = ∑  (1 - y_i) * (1 - F(p_i))
          i=1
           n
      FP = ∑  (1 - y_i) * F(p_i)
          i=1
           n
      FN = ∑  y_i * (1 - F(p_i))
          i=1

      where y_i represents the true label, p_i represents the predicted probability given by p_i = sigmoid(x_i) and
      F represents the a priori distribution for the threshold.

      The Score-Oriented loss is defined on the elements of the expected confusion matrix as follows

      loss = - score(TP,TN,FP,FN) + 1,

      where score represents the chosen skill score.

      Example
      if score = 'accuracy'
      then
      loss = - (TP + TN) / (TP + FN + TN + FP) + 1
    
    Authors: Guastavino S. & Marchetti F.

    References: https://arxiv.org/abs/2103.15522

    Usage:
     model.compile(loss=SOL(score = 'accuracy', distribution = 'uniform', mu = 0.5, delta = 0.1, mode = 'average'))

    Parameters
    ----------
    
    score : string, the chosen score used to build the loss. Implemented 
            choices are ['accuracy','precision','recall','specificity',
            'f1_score','tss','csi','hss1','hss2'].
    
    distribution : string, the a priori distribution for the threshold.
                   Implemented choices are ['uniform','cosine'].
             
    mu : scalar in (0,1) or list of scalars in (0,1). If the chosen 
         distribution is 'cosine', mu is the mean of the raised cosine 
         distribution. In the multiclass case, mu can be defined as a list of
         values, one for each one-vs-rest classification, so that
         len(mu) = number of classes. 
         If the the chosen distribution is 'uniform', this parameter is
         ignored.
    
    delta : scalar in (0,1). If the chosen distribution is 'cosine', then
            [mu-delta,mu+delta] is the support of the raised cosine
            distribution. In the multiclass case, delta can be defined as a
            list values, one for each one-vs-rest classification, so that
            len(delta) = number of classes. 
            If the the chosen distribution is 'uniform', this parameter is
            ignored.
    
    mode : string in ['average','weighted']. In the multiclass case, it 
           determines in which way the contributes of the one-vs-rest tasks
           are combined in a unique score. 
           If the problem is not multiclass, this parameter is ignored.
    
    
    """
        
    if score == 'accuracy':
        score = acc
    if score == 'precision':
        score = prec
    if score == 'recall':
        score = rec    
    if score == 'specificity':
        score = spec        
    if score == 'f1_score':
        score = f1s
    if score == 'tss':
        score = tss
    if score == 'csi':
        score = csi
    if score == 'hss1':
        score = hss1        
    if score == 'hss2':
        score = hss2 
        
    if distribution == 'uniform':
        distr = F_unif
        
    if distribution == 'cosine':
        if type(mu) is not list:
            distr = lambda x: F_cos(x,mu,delta)
        else:
            distr = [lambda x: F_cos(x,mu[j],delta[j]) for j in \
                     range(0,len(mu))]
            
    def SOL_(y_true, y_pred):
                
        y_true = tf.convert_to_tensor(y_true, tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, tf.float32)
        
        if y_pred.shape[1] == 1:   
            
            TN = tf.reduce_sum((1.-y_true)*(1.-distr(y_pred)))
            TP = tf.reduce_sum(y_true*distr(y_pred))
            FP = tf.reduce_sum((1.-y_true)*distr(y_pred))
            FN = tf.reduce_sum(y_true*(1.-distr(y_pred)))
        
            return -score(TN,FP,FN,TP) + 1.
        
        else:
            
            score_arr = []
            num_c0_arr = []
            
            
            for j in range(0,y_pred.shape[1]):
                                 
                y_pred_ = y_pred[:,j]
                y_true_ = y_true[:,j]
                
                if type(mu) is not list:

                    TN = tf.reduce_sum((1.-y_true_)*(1.-distr(y_pred_)))
                    TP = tf.reduce_sum(y_true_*distr(y_pred_))
                    FP = tf.reduce_sum((1.-y_true_)*distr(y_pred_))
                    FN = tf.reduce_sum(y_true_*(1.-distr(y_pred_)))
                
                else:
                
                    TN = tf.reduce_sum((1.-y_true_)*(1.-distr[j](y_pred_)))
                    TP = tf.reduce_sum(y_true_*distr[j](y_pred_))
                    FP = tf.reduce_sum((1.-y_true_)*distr[j](y_pred_))
                    FN = tf.reduce_sum(y_true_*(1.-distr[j](y_pred_)))
                    
                score_arr.append(score(TN,FP,FN,TP))

                if mode == 'weighted':
                    num_c0_arr.append(tf.cast(tf.shape(y_true_)[0],\
                                        tf.float32)-tf.reduce_sum(y_true_))
            
            score_arr = tf.stack(score_arr,axis=0)
            
            if mode == 'weighted':
                num_c0_arr = tf.stack(num_c0_arr,axis=0)
                final_score = tf.math.divide_no_nan(tf.reduce_sum\
                            (score_arr*num_c0_arr),tf.reduce_sum(num_c0_arr))
                
            if mode == 'average':
                final_score = tf.math.reduce_mean(score_arr)
            
            return -final_score + 1.
    
    return SOL_
