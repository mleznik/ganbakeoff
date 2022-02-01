from scipy.fft import fft, fftfreq
import numpy as np
import matplotlib.pyplot as plt
import wandb
import antropy as ant
#from scipy.special import softmax

def softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis] # necessary step to do broadcasting
    e_x = np.exp((z - s)*3)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis] # dito
    return e_x / div


def fft_evaluation(N,T,time_series, use_softmax=False, remove_0_bin=True):
        
    
    yf = fft(time_series)
    xf = fftfreq(N, T)[:N//2]
    
    abs_val = 2.0/N * np.abs(yf[:,0:N//2])
    
    if remove_0_bin:
        abs_val = np.concatenate((np.zeros(shape=[abs_val.shape[0],1]), abs_val[:,1:]),axis=1)
    
    if use_softmax:
        normalized = softmax(abs_val)
    else:
        summe = np.sum(abs_val,axis=1)
        
        indices = ~(summe == 0.0)
        
        summe = summe[indices]
        
    
        summe_array = np.stack([[summe[i]] * (N//2)  for i in range(summe.shape[0])],axis=0)
    
        normalized = abs_val[indices] / summe_array
              
    
    return abs_val[indices], normalized

def autocorrelation(time_series):
    def autocorr(x):
        result = np.correlate(x, x, mode='full')
        return result[result.size//2:]
    return np.apply_along_axis(autocorr,1,time_series)

def correlation(time_series):
    
    res = []
    for time_ser in time_series:
        coef = np.corrcoef(time_ser.T)
        
        if not np.isnan(coef).any():
            res.append(coef)
        
    return np.array(res)

def mean(time_series):
    return np.mean(time_series,axis=1)

def variance(time_series):
    return np.var(time_series, axis=1)

def cdf(time_series, plot=False):
    
    num_examples, num_steps, num_channels = time_series.shape
    
    
    fig,axs = plt.subplots(num_examples,num_channels, figsize=(num_examples*5, 10))
    
    for i in range(num_examples):
        for j in range(num_channels):
            X2 = np.sort(time_series[i,:,j])
            F2 = np.array(range(num_steps))/float(num_steps)
            
            if num_channels > 1:
                axs[i,j].plot(X2, F2)
            else:
                axs[i].plot(X2,F2)
            
    if not plot:
        plt.close(fig)
    return fig

def entropy(time_series):
       
    save_dict = dict()
    
    function_dict = {
        "app_entropy": ant.app_entropy,
    }
    

    for key in function_dict.keys():
        save_dict[key] = []
        for time_ser in time_series:
            ret_list = []
            for chann in range(time_series.shape[2]):

                x = time_ser[:,chann]
                val = function_dict[key](x)
                ret_list.append(val)
            
            if not np.isnan(ret_list).any():
                save_dict[key].append(ret_list)
    return save_dict

def evaluation_pipeline(time_series, plot=False, remove_0_bin=True, num_frequencies=1):
    
    return_dict = dict()
    
    ### FFT
    
    abs_values = []
    normalized_values = []
    
    for i in range(time_series.shape[2]):
        a, n = fft_evaluation(time_series.shape[1], 1/time_series.shape[1], time_series[:,:,i], remove_0_bin=remove_0_bin)
        
        abs_values.append(a)
        normalized_values.append(n)
        
    eval_values = [normalized_values[i][np.repeat(np.arange(normalized_values[i].shape[0]),num_frequencies).reshape(-1,num_frequencies),np.argsort(-normalized_values[i])[:,:num_frequencies]] for i in range(time_series.shape[2])]
        
    
    return_dict["fft"] = {"abs": np.array(abs_values), "norm": np.array(normalized_values), "eval": np.array(eval_values)}
    
    ### correlation
    
    return_dict["correlation"] = {"autocorr": autocorrelation(time_series), "cross-corr": correlation(time_series)}
    
    return_dict["entropy"] = entropy(time_series)
    
    return return_dict
    
    
    