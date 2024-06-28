import os, random, time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K
from scipy.special import polygamma, loggamma
from lifelines.utils import concordance_index

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    tf.random.set_seed(seed)
    
class config:
    seed = 42
    device = "cuda:0" 

def brier_score(y, d, mu, interval=0.1):

    time_range = np.arange(0, max(y), interval)

    temp_df = pd.DataFrame()
    temp_df['y'], temp_df['d'], temp_df['mu'] = y, d, mu
    temp_df.sort_values(by=['y'])
    y, d, mu = np.array(temp_df['y']), np.array(temp_df['d']), np.array(temp_df['mu'])
    
    # y, d

    if 0 in d: y_unique_pos, tie_count_pos = np.delete(np.unique(y*d, return_counts=True), 0, 1)
    else: y_unique_pos, tie_count_pos = np.unique(y*d, return_counts=True)
    Mi_pos = np.array([yi >= y_unique_pos for yi in y])

    # y, 1-d

    if 1 in d: y_unique_neg, tie_count_neg = np.delete(np.unique(y*(1-d), return_counts=True), 0, 1)
    else: y_unique_neg, tie_count_neg = np.unique(y*(1-d), return_counts=True)
    Mi_neg = np.array([yi >= y_unique_neg for yi in y])

    Ak_pos = tie_count_pos / np.sum(Mi_pos, axis=0)
    Ak_neg = tie_count_neg / np.sum(Mi_neg, axis=0)
    Gx_pos = np.exp(Mi_pos@(np.log(1-Ak_pos+1e-7)))
    Gx_neg = np.exp(Mi_neg@(np.log(1-Ak_neg+1e-7)))  
    Gx_haz = Mi_pos@ (tie_count_pos/(mu@Mi_pos))

    Gt0 = np.ones(np.shape(time_range))
    for yi in y_unique_neg:
        Gt0[time_range>=yi] = Gx_neg[y==yi][-1]
    St0  = np.ones(np.shape(time_range))
    Lt0 = np.zeros(np.shape(time_range))
    for yi in y_unique_pos:
        St0[time_range>=yi] = Gx_pos[y==yi][-1]
        Lt0[time_range>=yi] = Gx_haz[y==yi][-1]

    BS0, BSc = [], []
    for t in range(len(time_range)):
        ot = np.where(y > time_range[t], 1, 0)
        wt = (1-ot)*d/Gx_neg + ot/Gt0[t]
        Stc = np.exp(-Lt0[t])        
        BS0.append(np.mean(((ot-St0[t])**2)*wt))
        BSc.append(np.mean(((ot-Stc**mu)**2)*wt))
        
    ibs0,ibsc0 = [], []        
    for idx in range(len(time_range)-1):
        ibs0.append( np.diff(time_range)[idx]*((BS0[idx]+BS0[idx+1])/2))
        ibsc0.append(np.diff(time_range)[idx]*((BSc[idx]+BSc[idx+1])/2))
     
    ibs  = sum(ibs0) /(max(time_range)-min(time_range))  
    ibsc = sum(ibsc0)/(max(time_range)-min(time_range))
    
    result = {'t0' : time_range, 'Reference' : BS0, 'Brier Score' : BSc, 'Reference_ibs' : ibs, 'IBS' : ibsc}        
    return result

def cumul_hazard(y, d, xb, zv):
    if 0 in d: y_unique, tie_count = np.delete(np.unique(y*d, return_counts=True), 0, 1)
    else: y_unique, tie_count = np.unique(y*d, return_counts=True)
    Mi = np.array([yi >= y_unique for yi in y])
    res = Mi@(tie_count/(np.exp(xb+zv).T@Mi)[0])
    return res

def d1_lambda(lam, y_sum, mu_sum):
    out = (
        polygamma(0, (y_sum+(1/lam))) - polygamma(0, (1/lam)) + np.log((1/lam)) + 1 
        - np.log(mu_sum+(1/lam)) - (y_sum+(1/lam))/(mu_sum+(1/lam))
    )
    out = - (1/lam**2) * out
    return np.sum(out)

def d2_lambda(lam, y_sum, mu_sum):
    out = (
        polygamma(1, (y_sum+(1/lam))) - polygamma(1, (1/lam)) + lam - 1/(mu_sum+(1/lam)) 
        - (mu_sum-y_sum)/((mu_sum+(1/lam))**2)
    )
    out = (1/lam**4) * out
    #out = (2/lam**3) * out
    return np.sum(out)

def gf_hlik_loss(mean_model, X, Z, y, d, d_sum, cluster_size, lam, disp):
    
    # cluster size is a vector of cluster sizes
    # canonicalizer is a vector of ci(lam; d)
    # X, Z, y, d are from the mini-batch
    # d_sum and cluster_size are from the whole train set
    
    xb, zv = mean_model([X, Z])
    mu = K.flatten(K.exp(xb+zv))

    sort_index = K.reverse(tf.nn.top_k(y, k=np.shape(y)[0], sorted=True).indices, axes=0)
    y_sort = K.gather(reference=y,  indices=sort_index)
    d_sort = K.gather(reference=d,  indices=sort_index)
    y_pred = K.gather(reference=mu, indices=sort_index)
    
    tie_count = np.unique(y_sort[d_sort==1], return_counts=True)[1]
    tie_count = tf.convert_to_tensor(tie_count.reshape(1, -1), dtype=tf.float32)    
    # if 0 in d_sort: tie_count = K.expand_dims(tf.unique_with_counts(y_sort*d_sort).count[1:],0)
    # else: tie_count = K.expand_dims(tf.unique_with_counts(y_sort*d_sort).count,0)

    ind_mat = tf.cast(K.expand_dims(y_sort,0) == K.expand_dims(np.unique(y_sort), 1), tf.float32)
    cum_haz = tf.linalg.band_part(K.ones((K.shape(ind_mat)[0], K.shape(ind_mat)[0])), 0, -1)@ind_mat@K.expand_dims(y_pred)
    hlik= (K.expand_dims(d,0)@(xb+zv) - tie_count@K.expand_dims(K.log(cum_haz[ind_mat@K.expand_dims(y_pred*d_sort)!=0]))) / np.shape(y)[0]
    
    # ind_mat@K.expand_dims(y_pred*d_sort) == 0 은 어떤 y의 값에 대해 모든 관측된 tie가 censoring 또는 event지만 예측값이 0
    # 반대로 != 0 이면 관측된 모든 tie 중에서 양수의 예측값을 갖는 event가 존재하는 것들
    # tie_count = event 중에서 unique 한 관측값에 대해서만 카운트한 것
    # 즉, 차이가 난다면 예측값이 0인 event가 존재한다는 뜻! 하지만 K.exp(xb+zv)는 양수여야 함!
    # 그럳하면 y_pred 가 nan이 나와서 censoring인데도 nan * 0 = nan 나와서 dimension 에러 발생했을 것임!
    
    # f(v)
    # v = mean_model.weights[-1]
    # hlik += K.sum((v - K.exp(v) - K.log(lam))/lam - tf.math.lgamma(1/lam)) / K.sum(cluster_size)
    hlik += K.sum((zv/lam - K.exp(zv)/lam)/(Z@cluster_size)) / np.shape(y)[0]
    
    # canonicalizer    
    # hlik += K.sum( -(d_sum+1/lam)*K.log(d_sum+1/lam) + d_sum+1/lam + tf.math.lgamma(d_sum+1/lam) ) / K.sum(cluster_size)
    if disp:
        hlik += K.sum((
            - K.log(lam)/lam - tf.math.lgamma(1/lam)
            + Z@(-(d_sum+1/lam)*K.log(d_sum+1/lam) + d_sum+1/lam + tf.math.lgamma(d_sum+1/lam))) 
            / (Z@cluster_size)
        ) / np.shape(y)[0]
    
    loss = - hlik
    return loss

def gf_update_params(mean_model, X, Z, y, d, d_sum, cluster_size, lam, optimizer, disp):  
    with tf.GradientTape() as tape:
        loss = gf_hlik_loss(mean_model, X, Z, y, d, d_sum, cluster_size, lam, disp)
    if disp:
        gradients = tape.gradient(loss, mean_model.trainable_weights+[lam])
        optimizer.apply_gradients(zip(gradients, mean_model.trainable_weights+[lam])) 
    else:
        gradients = tape.gradient(loss, mean_model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, mean_model.trainable_weights))
    return loss

def gf_train_one_epoch(mean_model, train_batch, d_sum, cluster_size, lam, optimizer, disp):    
    losses = [] 
    for step, (X_batch, Z_batch, y_batch, d_batch) in enumerate(train_batch):
        loss = gf_update_params(mean_model, X_batch, Z_batch, y_batch, d_batch, d_sum, cluster_size, lam, optimizer, disp)
        losses.append(loss)
    return losses

def adjust_wts(wts, lam):    
    v_adjs = - np.log(np.mean(np.exp(wts[-1])))
    # wts[-2] = wts[-2] - v_adjs
    wts[-1] = wts[-1] + v_adjs        
    return wts

def find_rand_var(mean_model, X, Z, y, d, lam_init=0.5, maxiter=1000, threshold=1e-5):
    lam = lam_init
    xb, zv = mean_model([X, Z])
    mu_sum = Z.T@(cumul_hazard(y, d, xb, zv)*np.exp(xb).T[0])
    y_sum = (Z.T@y).reshape(-1,1)
    for _ in range(maxiter):
        update = d1_lambda(lam, y_sum, mu_sum)/d2_lambda(lam, y_sum, mu_sum)
        lam -= update
        if lam < 0:
            lam.assign(1e-18)
        if np.abs(update) < threshold:
            break
    return np.float32(lam)

def gf_train_model(
    mean_model, train_batch, train_data, validation_data, optimizer, lam_init, 
    pretrain=20, patience=20, max_epochs=1000, adjust=True, disp=True, disp_update = 1, 
    monitor = 'loss', pretrain_optimizer = None, wts_init=None, seed=42):

    X_train, Z_train, y_train, d_train = train_data
    X_valid, Z_valid, y_valid, d_valid = validation_data
    N_train, N_valid = np.shape(y_train)[0], np.shape(y_valid)[0]
    
    cluster_size_train = np.diag(Z_train.T@Z_train).reshape(-1,1)
    cluster_size_valid = np.diag(Z_valid.T@Z_valid).reshape(-1,1)
    y_sum_train, y_sum_valid = (Z_train.T@y_train).reshape(-1,1), (Z_valid.T@y_valid).reshape(-1,1)
    d_sum_train, d_sum_valid = (Z_train.T@d_train).reshape(-1,1), (Z_valid.T@d_valid).reshape(-1,1)    
       
    if wts_init is not None:
        mean_model.set_weights(wts_init)
    lam = tf.Variable(lam_init, name='lam', trainable=True, constraint=lambda x: tf.clip_by_value(x, 1e-18, np.infty))

    train_losses, train_mse, train_cindex, train_bscore = [np.zeros(max_epochs) for _ in range(4)]
    valid_losses, valid_mse, valid_cindex, valid_bscore = [np.zeros(max_epochs) for _ in range(4)]
    lam_history, compute_time = [np.zeros(max_epochs) for _ in range(2)]

    if pretrain_optimizer is None: pretrain_optimizer = optimizer
    for epoch in range(pretrain):
        train_loss = gf_train_one_epoch(mean_model, train_batch, d_sum_train, cluster_size_train, lam, pretrain_optimizer, disp=False)
    # min_valid_loss = gf_hlik_loss(mean_model, X_valid, Z_valid, y_valid, d_valid, d_sum_valid, cluster_size_valid, lam, disp)
    
    min_valid_measure = np.infty
    patience_count = 0
    
    temp_start = time.time()
    for epoch in range(max_epochs):
        if epoch%disp_update == 0: 
            train_loss = gf_train_one_epoch(mean_model, train_batch, d_sum_train, cluster_size_train, lam, optimizer, disp)
            if disp=='newton':
                lam.assign(find_rand_var(mean_model, X_train, Z_train, y_train, d_train, lam_init=float(lam), maxiter=1000, threshold=1e-5))
            # print(float(lam))
        else: train_loss = gf_train_one_epoch(mean_model, train_batch, d_sum_train, cluster_size_train, lam, optimizer, disp=False)
        # train_loss = gf_train_one_epoch(mean_model, train_batch, d_sum_train, cluster_size, lam, optimizer, disp)
        # xb_train, zv_train = mean_model([X_train, Z_train])
            # mu_sum_train = Z_train.T@(cumul_hazard(y_train, d_train, xb_train, zv_train)*np.exp(xb_train).T[0])
            # lam.assign(lam - d1_lambda(lam, y_sum_train, mu_sum_train)/d2_lambda(lam, y_sum_train, mu_sum_train))
            # if lam < 0:
            #     lam.assign(1e-18)
        wts = mean_model.get_weights()
        if adjust:
            wts = adjust_wts(wts, lam)
            mean_model.set_weights(wts)
        valid_loss = gf_hlik_loss(mean_model, X_valid, Z_valid, y_valid, d_valid, d_sum_valid, cluster_size_valid, lam, disp=True)
        
        compute_time[epoch] = time.time() - temp_start
        
        xb_train, zv_train = mean_model([X_train, Z_train])
        xb_valid, zv_valid = mean_model([X_valid, Z_valid])
        mu_train = np.exp(xb_train+zv_train)
        mu_valid = np.exp(xb_valid+zv_valid)
        
        train_mse[epoch] = np.mean((y_train-mu_train.T)**2)
        valid_mse[epoch] = np.mean((y_valid-mu_valid.T)**2)
        
        train_losses[epoch] = np.mean(train_loss)
        valid_losses[epoch] = np.float32(valid_loss)[0][0]
        
        train_cindex[epoch] = concordance_index(y_train, -mu_train, event_observed = d_train)
        valid_cindex[epoch] = concordance_index(y_valid, -mu_valid, event_observed = d_valid)
        
        train_bscore[epoch] = brier_score(y_train, d_train, mu_train)['IBS']
        valid_bscore[epoch] = brier_score(y_valid, d_valid, mu_valid)['IBS']
        
        lam_history[epoch] = np.float32(lam)
        
        valid_measures = {
            'mse'    : valid_mse[epoch],
            'loss'   : valid_losses[epoch],            
            'cindex' : (1-valid_cindex[epoch]),
            'bscore' : valid_bscore[epoch]
        }
        
        valid_measure = valid_measures[monitor]        
        if valid_measure > min_valid_measure:
            patience_count += 1
            if patience_count == patience: break
        else: 
            min_valid_measure = valid_measure
            patience_count = 0

    res = {
        "wts" : wts,
        "lam" : np.float32(lam),
        "train_mse"   : train_mse[:(epoch+1)],
        "valid_mse"   : valid_mse[:(epoch+1)],
        "train_loss"  : train_losses[:(epoch+1)],
        "valid_loss"  : valid_losses[:(epoch+1)],
        "train_cindex": train_cindex[:(epoch+1)],
        "valid_cindex": valid_cindex[:(epoch+1)],
        "train_bscore": train_bscore[:(epoch+1)],
        "valid_bscore": valid_bscore[:(epoch+1)],
        "lam_history" : lam_history[:(epoch+1)],
        "compute_time": compute_time[:(epoch+1)]
    }
    
    return res

def coxph_loss(mean_model, X, y, d):

    mu = K.exp(K.transpose(mean_model([X])))

    sort_index = K.reverse(tf.nn.top_k(y, k=np.shape(y)[0], sorted=True).indices, axes=0)
    y_sort = K.gather(reference=y,  indices=sort_index)
    d_sort = K.gather(reference=d,  indices=sort_index)
    y_pred = K.gather(reference=mu, indices=sort_index)

    yd = y_sort*d_sort
    tie_count = tf.cast(tf.unique_with_counts(
            tf.boolean_mask(yd, tf.greater(yd, 0))
        ).count, dtype = tf.float32)


    ind_matrix = K.expand_dims(y_sort, 0) - K.expand_dims(y_sort, 1)
    ind_matrix = K.equal(x = ind_matrix, y = K.zeros_like(ind_matrix))
    ind_matrix = K.cast(x = ind_matrix, dtype = tf.float32)
    
    time_count = K.cumsum(tf.unique_with_counts(y_sort).count)
    time_count = K.cast(time_count - K.ones_like(time_count), dtype = tf.int32)

    ind_matrix = K.gather(ind_matrix, time_count)
    
    tie_haz = y_pred * d_sort
    tie_haz = K.dot(ind_matrix, K.expand_dims(tie_haz)) 
    event_index = tf.math.not_equal(tie_haz,0) 
    tie_haz = tf.boolean_mask(tie_haz, event_index)
    
    tie_risk = K.log(y_pred) * d_sort
    tie_risk = K.dot(ind_matrix, K.expand_dims(tie_risk))
    tie_risk = tf.boolean_mask(tie_risk, event_index)
    
    cum_haz = K.dot(ind_matrix, K.expand_dims(y_pred))
    cum_haz = K.reverse(tf.cumsum(K.reverse(cum_haz, axes = 0)), axes = 0)
    cum_haz = tf.boolean_mask(cum_haz, event_index)

    plik = tf.math.reduce_sum(tie_risk)-tf.math.reduce_sum(tie_count*tf.math.log(cum_haz))    
                   
    return -plik