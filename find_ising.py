import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
import tensorflow as tf
from tqdm import trange
from matplotlib import pyplot as plt
import itertools
import bottleneck as bn
import scipy.io
from functions import *
import os

plt.rcParams.update({'font.size': 20})


# __________________________________________________________________________
### read command line arguments ###
import argparse
parser = argparse.ArgumentParser(
                    prog='find_parameters.py',
                    description='Find parameters of Ising model')

parser.add_argument('-d',type=int, help='number of variables', required=True)
parser.add_argument('-label', help='class of ising model', required=True)
parser.add_argument('-n_samples',type=int, help='number of samples', default=10000)
parser.add_argument('-n_thermalize',type=int, help='number of steps to thermalize', default=50000)
parser.add_argument('-n_step_thermalize',type=int, help='inside the step function steps to thermalize', default=50)
parser.add_argument('-n_step_compute_f',type=int, help='inside the step function samples to compute f', default=20)
parser.add_argument('-n_steps',type=int, help='number of steps in the evolution', default=20000)
parser.add_argument('-n_step_decorrelate',type=int, help='inside the step function steps to decorrelate inside compute f', default=2)
parser.add_argument('-adam_step',type=float, help='step size', default=1e-3)


args = parser.parse_args()
d = args.d
label=args.label
n_samples = args.n_samples
n_thermalize = args.n_thermalize
adam_step = args.adam_step
n_step_thermalize =args.n_step_thermalize
n_step_compute_f = args.n_step_compute_f
n_step_decorrelate = args.n_step_decorrelate
n_steps = args.n_steps
# __________________________________________________________________________
# folder_bin = '../../../../scratch/network/csarra/allen_saved/binary/'
folder_bin = ''
f_data = np.loadtxt(f'{folder_bin}f_data_{d}_{label}.dat')

if not os.path.isfile(f'{folder_bin}f_dataERR_{d}_{label}.dat'):
    errs = np.ones(len(f_data))*1e-5
else:
    errs = np.loadtxt(f'{folder_bin}f_dataERR_{d}_{label}.dat')


if not os.path.isfile(f'{folder_bin}q_{d}_{label}.dat'):
    print('no q found, start from independent')
    q1 = np.log((1-f_data[:d])/(1+f_data[:d]))/2*d
    q = np.hstack((q1, [0]*int(d*(d-1)/2)))
else:
    print('q loaded from file')
    q = np.loadtxt(f'{folder_bin}q_{d}_{label}.dat')

q= tf.Variable(q, dtype=tf.float64)


plot_sum = True
if not os.path.isfile(f'{folder_bin}_histosum_{d}_{label}.dat'):
    plot_sum = False
else:
    zsum = np.loadtxt(f'{folder_bin}_histosum_{d}_{label}.dat')
    zsums = np.loadtxt(f'{folder_bin}_histosums_{d}_{label}.dat')


#---------------------2. Find parameters
h,J = get_hJ(q,d)
h = tf.convert_to_tensor(h,dtype=tf.float64)
J= tf.convert_to_tensor(J,dtype=tf.float64)


x = np.random.choice([-1,1], (n_samples,d))
x = tf.convert_to_tensor(x,dtype=tf.float64)

for k in range(5):
    x, energy_history = thermalize(x,J,h, NumSteps=n_thermalize , crange =range)

x, f_model = compute_f(x,J,h,StepsTherm=10,NumSteps=50, crange=lambda NumSteps, dtype: trange(NumSteps))


f_model_history =[]
q_history=[]
mre_history=[]
m2conn_diff =[]

f_data = tf.convert_to_tensor(f_data, tf.float64)

# ---------------------- THERMALIZE
optimizer = tf.optimizers.Adam(adam_step)

sorted_different_indeces = np.argsort(abs((np.array(f_data - f_model)))/errs)[::-1]
i = sorted_different_indeces[:1]
print(i,(abs((np.array(f_data - f_model)))/errs)[i])


for t in trange(n_steps):
    h,J = get_hJ(q,d)
    h = tf.convert_to_tensor(h,dtype=tf.float64)
    J= tf.convert_to_tensor(J,dtype=tf.float64)
    x,f_model = step(q,x,J,h,f_data, n_step_thermalize, n_step_compute_f,n_step_decorrelate, optimizer)
    
    f_model_history.append(f_model.numpy()[i])
    q_history.append(q.numpy()[i]) 
       
    mre_history.append(tf.reduce_sum((f_model-f_data)**2).numpy())
    np.savetxt(f'{folder_bin}q_{d}_{label}.dat', np.array(q))
    
    
    #__________________________________________________________________________________
    ## -------------------------- PLOTS -----------------------
    ## --------------------------------------------------------
    if ((t)%int(n_steps/20)==0):
        x, energy_history = thermalize(x,J,h, NumSteps=1000 , crange =range)
        
        plt.figure(figsize=(15,10))
        plt.subplot(2,3,1)
        plt.plot(mre_history[:])
        plt.xlabel('epochs')
        plt.ylabel(r'||$f_{data} - f_{model}||^2$')
        
        # 
        # ---------------- Plot parameters evolution ------------------
        plt.subplot(2,3,2)
        plt.plot(np.array(f_model_history))
        plt.axhline(np.array(f_data)[i])
        plt.axhline(3*errs[i]+np.array(f_data)[i])
        plt.axhline(-3*errs[i]+np.array(f_data)[i])
        plt.title(f'Averages {i}')
        plt.xlabel('epochs')
        plt.ylabel(r'$f_{model}/f_{data}$')
        
        plt.subplot(2,3,3)
        plt.plot(np.array(q_history)[:])
        plt.title(f'Parameters {i}')
        plt.xlabel('epochs')
        plt.ylabel('q')
        plt.tight_layout()
          
        plt.subplot(2,3,4)
        plt.plot(abs(f_data.numpy()-f_model.numpy()))
        plt.plot(errs)
        plt.plot(errs*3)        
        plt.title((abs(f_data.numpy()-f_model.numpy())>3*errs).sum())

        plt.subplot(2,3,5)
        m2conn_data = np.array(f_data[d:] - np.array(f_data[:d][:,None]*f_data[:d][None,:])[np.triu_indices(d,1)])
        m2conn_model = np.array(f_model[d:]-np.array(f_model[:d][:,None]*f_model[:d][None,:])[np.triu_indices(d,1)])
        plt.scatter(m2conn_data[::10] ,m2conn_model[::10], s=5)
        xmin, xmax = plt.gca().get_xlim()
        xx = np.linspace(xmin,xmax,100)
        plt.plot(xx,xx, color='red')
        plt.title(r'$\langle x_i x_j\rangle_{conn}$')
        
        plt.subplot(2,3,6)
        if plot_sum==True:
            s=[]
            for _ in range(10):
                x, energy_history = thermalize(x,J,h, NumSteps=20 , crange =range)
                s.append(x.numpy().sum(1))
            s=np.array(s).flatten()
            h0,h1 = np.unique(s, return_counts=True); h1=h1/h1.sum()
            plt.plot(h0,h1, '-', label='model', color='red')
            plt.plot(zsum,pzsum, '-', label='data', color='black')
            plt.xlabel('sum')
            
        plt.tight_layout()
        plt.savefig(f"{folder_bin}evolution_{d}_{label}.pdf", format="pdf", bbox_inches="tight")
        plt.show()
        
        
    if ((abs(f_data.numpy()-f_model.numpy())>3*errs).sum()==0): 
        print('Converged')
        break