import anndata as ad
import h5py
import json
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
import tensorflow as tf
from tqdm import trange

# ___________________________________________________________________________________________
DATA_PATH = '' # TODO: enter here the data path
# ___________________________________________________________________________________________

def get_hJ(q,d):
    h = q[:d]
    J = np.zeros((d,d))
    J[np.triu_indices(d,1)] = q[d:]
    J = (J + J.T)/2
    return h,J

# NOTE: q is related to the Ising model parameters through a factor d
def get_H(x,J,h):
    num_part,d = x.shape
    return tf.reduce_sum(x*h + x * tf.reduce_sum(J[None,:,:] * x[:,:,None],1),1)/d


@tf.function
def one_step_MC(x, h,J,Hx):
    num_part,d = x.shape
    indices = tf.random.uniform([num_part], 0,d, dtype=tf.int64)
    dy = tf.one_hot(indices,d,on_value=-1,off_value=1, dtype=tf.float64)
    y = x*dy
    Dh_h = tf.gather(h, indices)
    Dh_J = tf.reduce_sum(tf.gather(J, indices, batch_dims=0)*x,1)
    DH = -2 * tf.gather(x,indices, batch_dims=1)*(Dh_h + 2*Dh_J)/d
    Hy = Hx+DH
    r = tf.math.log(tf.random.uniform([num_part], 0, 1, dtype=tf.float64))
    x = y*tf.cast(r<-DH, dtype=tf.float64)[:,None] + x*tf.cast(r>=-DH, dtype=tf.float64)[:,None]
    Hx = Hy*tf.cast(r<-DH, dtype=tf.float64) + Hx*tf.cast(r>=-DH, dtype=tf.float64)
    return x, Hx


def thermalize(x, J,h, NumSteps = 1000, crange=tf.range):
    num_part,d = x.shape
    Hx = get_H(x,J,h)
    energy_history = tf.reduce_mean(Hx, keepdims=True)
    for _ in crange(NumSteps):
        x,Hx = one_step_MC(x, h,J,Hx)
        energy_history = tf.concat((energy_history,tf.reduce_mean(Hx, keepdims=True) ),0)  
    return x, energy_history


def step(q,x,J,h,f_data, n_step_thermalize, n_step_compute_f,n_step_decorrelate, optimizer):
    x, energy_history = thermalize(x,J,h,NumSteps=n_step_thermalize)
    x, f = compute_f(x,J,h,StepsTherm=n_step_decorrelate,NumSteps=n_step_compute_f) 
    delta_q = f-f_data
    optimizer.apply_gradients(zip([-delta_q],[q]))
    return x, f


def compute_f(x, J,h, NumSteps=1000, StepsTherm=5, crange=tf.range):
    num_part,d = x.shape
    num_par = int(d+ d*(d-1)/2)

    f_data = tf.zeros(num_par, dtype=tf.float64)
    Hx = get_H(x,J,h)
    mask_upper = tf.linalg.band_part(tf.ones((d,d)) - tf.eye(d), 0, -1)
    for n in crange(NumSteps, dtype = tf.float64):
        for _ in tf.range(StepsTherm):
            x,Hx = one_step_MC(x, h,J,Hx)
        f_data_partial = tf.concat((tf.reduce_mean(x,0), tf.boolean_mask(tf.reduce_mean(x[:,:,None]*x[:,None,:],0), mask_upper)),0)
        f_data = (n*f_data + f_data_partial)/(n+1)  
    return x, f_data


def get_f_data(data,d):
    datamean = data.mean(0)
    data_mom2 = ((data[:,:,None]*data[:,None,:])[:, np.triu_indices(d,1)[0], np.triu_indices(d,1)[1]]).mean(0)
    f_data = np.concatenate((datamean, data_mom2))  
    return f_data


def get_errorfdata(data,d):
    f =[]
    for _ in range(10):
        idx = np.random.choice(range(len(data)), int(len(data)/2), replace=False)
        f.append(get_f_data(data[idx],d))
    f=np.array(f)
    error = f.std(0)
    return error


def initialize_q(f_data,d):
    corr_conn = f_data[d:] - np.array((f_data[:d][:,None]*f_data[:d][None,:]))[np.triu_indices(d, k=1)[0],np.triu_indices(d, k=1)[1]]
    q=np.zeros(len(f_data))
    q[d:] = -corr_conn
    q[:d] = - f_data[:d]
    q*=d
    return q


def load_data(path):
    data_tot = scipy.io.loadmat(path)
    count = data_tot['RNAcount'].astype(np.float64)
    count_right = count[:,10:]
    # count_right/=count_right.sum(1)[:,None] # ***** comment this line for original  *****
    num_cells = count_right.shape[0]
    num_RNAs = count_right.shape[1]
    totnum = np.sum(count_right, axis=0)
    totnum_ord = np.sort(totnum)[::-1]
    rna = np.argsort(totnum)[::-1]  # RNA names ordered from the most frequent
    return count_right, rna


def get_points_fdata(indices, count,rna, d):
    points_data = count[:, rna[indices]]
    points_data = np.sign(points_data - points_data.mean(0))
    A = np.cov(points_data.T, bias=True)+points_data.mean(0)[:,None]*points_data.mean(0)[None,:]
    f_data = np.hstack((points_data.mean(0), A[np.triu_indices(d, k=1)]))
    f_data = tf.convert_to_tensor(f_data,dtype=tf.float64)   
    return points_data, f_data


# function that given a state and q returns the corresponding minimum
def get_minimum(x0, q):
    d = len(x0)

    x = np.copy(x0)
    actual_energy=[]
    h,J = get_hJ(q,d)
    actual_energy.append(get_H(x0[None,:],J,h)[0])

    for _ in range(100000):  #steps allowed to find a minimum  
        xx = np.repeat(x[None,:],d,0) 

        num_part,d = xx.shape
        indices_sel = range(num_part)
        Dh_h = tf.gather(h, indices_sel)
        Dh_J = tf.reduce_sum(tf.gather(J, indices_sel, batch_dims=0)*xx,1)
        DH = -2 * tf.gather(xx,indices_sel, batch_dims=1)*(Dh_h + 2*Dh_J)/d

        DH = np.array(DH)
        if DH.min()<0:
            isel = np.argmin(DH)
            x[isel]*=-1
            actual_energy.append(actual_energy[-1]+DH[isel])
        else:     
            return x

        
        
# ___________________________________________________________________________________________
#-----------------------------------  Small systems (exact averages instead of MC simulation)
def find_parameters(d,q, f_data, f_model, adam_step = 1e-1, num_steps=20000):
    optimizer = tf.optimizers.Adam(adam_step)
    @tf.function
    def step():
        f_model= get_f_model1(d,q) 
        delta_q = f_model-f_data
        optimizer.apply_gradients(zip([-delta_q],[q]))
        return f_model
    f_model_history =[]
    q_history=[]
    for _ in range(num_steps):
        f_model = step()
        f_model_history.append(f_model)
        q_history.append(q.numpy())    
        if (tf.reduce_max(abs(f_model-f_data)) < 1e-10):break
    return f_model_history, q_history

def get_p(num_var):
    x_vals =np.array([-1,1])
    if num_var == 1: arr=x_vals
    if num_var>1:
        arr = np.array([np.tile(x_vals, 2), np.repeat(x_vals, 2)]).T
    for i in range(num_var-2):
        v1 = np.tile(arr,(2,1))
        v2 = np.repeat(x_vals, len(arr))[:,None]
        arr = np.hstack((v1,v2))   
    arr[arr==0]=-1
    return arr

def get_f_model1(d,q):
    all_points = get_p(d)
    mixed = ((all_points[:,None,:]*all_points[:,:,None])[:,np.triu_indices(d, k=1)[0],np.triu_indices(d, k=1)[1]])
    v_points = np.hstack((all_points,mixed))
    h =tf.reduce_sum(v_points*q,1)/d
    p = tf.exp(-h); p/=tf.reduce_sum(p)
    f_model = tf.reduce_sum(v_points*p[:,None],0)
    return f_model

   
def get_qsmall(data, d, num_steps=5000):
    f_data = get_f_data(data,d)
    # ----  Monasson parameters
    mu = data.mean(0)
    c = np.cov(data.T)
    h,J = monasson_hj(mu,c)
    q_mon = np.hstack((-h, -J[np.triu_indices(d,1)]))*d
    q = tf.convert_to_tensor(q_mon)
    f_model = get_f_model1(d,q)
    
    # -------------------------- FIND q
    for adam_step in [1e-2, 1e-3, 1e-4]:
        q = tf.Variable(q, dtype = tf.float64)
        f_model_history, q_history =find_parameters(d,q, f_data, f_model, adam_step = adam_step, num_steps=num_steps)
        q = q_history[-1]
        f_model = get_f_model1(d,q)
        print(tf.reduce_max(abs(f_model - f_data)))
    
    # -------------------------- plot
    plt.figure(figsize=(8,4))

    plt.subplot(1,2,1)
    plt.plot((np.array(f_model_history)**2).sum(1))
    plt.xlabel('epochs')
    plt.ylabel(r'$||f_{data} - f_{model}||^2$')
    
    plt.subplot(1,2,2)
    plt.scatter(f_data[:d],f_model[:d], label= r'$\langle x \rangle$')
    plt.scatter(f_data[d:],f_model[d:], label= r'$\langle x_i x_j \rangle$')
    xmin,xmax = plt.gca().get_xlim()
    xx = np.linspace(xmin,xmax, 100)
    plt.plot(xx,xx)
    plt.xlabel(r'$f_{data}$')
    plt.ylabel(r'$f_{model}$')
    plt.legend()
    plt.tight_layout()
    
    return np.array(q)

def small_analysis(data, d, q):
    f_data = get_f_data(data,d)    
    # PROBABILITY
    all_points = get_p(d)
    mixed = ((all_points[:,None,:]*all_points[:,:,None])[:,np.triu_indices(d, k=1)[0],np.triu_indices(d, k=1)[1]])
    v_points = np.hstack((all_points,mixed))
    h =tf.reduce_sum(v_points*q,1)/d
    p = tf.exp(-h); p/=tf.reduce_sum(p)
    p=np.array(p)
    f_model = tf.reduce_sum(v_points*p[:,None],0)
    
    # model SUM DISTRIBUTION
    possible_sums = np.array(list(dict.fromkeys(all_points.sum(1))))
    prob_possible_sums = np.array([p[(all_points.sum(1)==possible_sums[i])].sum() for i in range(len(possible_sums))])
    # SUM
    plt.hist(data.sum(1),np.arange(-d,d,2), density=True, label='data');
    plt.scatter(1+possible_sums, prob_possible_sums/2, color='red', label='model')
    plt.plot(1+possible_sums, prob_possible_sums/2, color='red', ls='--')
    plt.xlabel('sum')
    plt.ylabel('P(sum)')
    plt.legend()

    # -------------  higher moments
    deltadata = data-f_data[:d]
    deltaall_points= np.array(all_points - f_model[:d])
    m2=[]
    m3=[]
    m4=[]
    for _ in trange(100):
        i,j,k,l = np.random.choice(10,4, replace=False)
        m2.append(((deltadata[:, [i,j]]).prod(1).mean(),    (deltaall_points[:,[i,j]].prod(1)*p).sum()))
        m3.append(((deltadata[:, [i,j,k]]).prod(1).mean(),  (deltaall_points[:,[i,j,k]].prod(1)*p).sum()))
        m4.append(((deltadata[:, [i,j,k,l]]).prod(1).mean(), (deltaall_points[:,[i,j,k,l]].prod(1)*p).sum()))
    m2=np.array(m2)
    m3=np.array(m3)    
    m4=np.array(m4)  

    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.scatter(m2[:,0], m2[:,1])
    xmin,xmax = plt.gca().get_xlim()
    xx = np.linspace(xmin,xmax, 100)
    plt.plot(xx,xx, color='black')
    plt.xlabel(r'$f_{data}$')
    plt.ylabel(r'$f_{model}$')
    plt.title(r'$\langle x_i x_j\rangle_{conn}$')

    plt.subplot(1,3,2)
    plt.scatter(m3[:,0], m3[:,1])
    xmin,xmax = plt.gca().get_xlim()
    xx = np.linspace(xmin,xmax, 100)
    plt.plot(xx,xx, color='black')
    plt.title(r'$\langle x_i x_j x_k\rangle_{conn}$')

    plt.subplot(1,3,3)
    plt.scatter(m4[:,0], m4[:,1])
    xmin,xmax = plt.gca().get_xlim()
    xx = np.linspace(xmin,xmax, 100)
    plt.plot(xx,xx, color='black')
    plt.title(r'$\langle x_i x_j x_k x_l\rangle_{conn}$')
    plt.tight_layout()
    plt.show()

    # find local minima
    minima =[]
    for cell in trange(len(all_points)):
        x = np.copy(all_points[cell])
        for _ in range(5000):
            prob_neigh = p[(np.abs(all_points-x)).sum(1)<=2]
            x1 = all_points[(np.abs(all_points-x)).sum(1)<=2][np.argmin(prob_neigh)]
            if np.all(x1==x):
                minima.append(x)
                break
            else: x = x1
    minima= np.array(minima)  
    tupled_lst = set(map(tuple, minima))
    lst = np.array(list(map(list, tupled_lst)))
    basin =[]
    for i in range(len(minima)):
        basin.append(np.where((minima[i] == lst).prod(1)==1)[0][0])
    basin = np.array(basin)
    print(len(lst))
    plt.imshow(lst)
    
    return p, lst, basin



def get_z(d,q):
    all_points = get_p(d)
    mixed = ((all_points[:,None,:]*all_points[:,:,None])[:,np.triu_indices(d, k=1)[0],np.triu_indices(d, k=1)[1]])
    v_points = np.hstack((all_points,mixed))
    h =tf.reduce_sum(v_points*q,1)/d
    p = tf.exp(-h); 
    z=tf.reduce_sum(p)
    return z
     
    
    
# # ___________________________________________________________________________________________
# #-------------------------------------------------- Approximated methods - Monasson         
# def monasson_hj(mu, c):
#     d = len(mu)
    
#     L = 1- mu**2
#     K = c/(L[:,None]*L[None,:])

#     # -------------------------------- find J
#     M = K*np.sqrt(L[:,None]*L[None,:])
#     M[range(d),range(d)] = 0

#     J_1l = np.sqrt(L[:,None]*L[None,:])*np.dot(M,np.linalg.inv(np.eye(d)+M))
#     J_2s = ((1/4*np.log((1+ K*(1+mu)[:,None]*(1+mu)[None,:])*(1+ K*(1-mu)[:,None]*(1-mu)[None,:])/(1- K*(1-mu)[:,None]*(1+mu)[None,:])/(1- K*(1+mu)[:,None]*(1-mu)[None,:]))))
#     J3 = K/(1-K**2*L[:,None]*L[None,:])
#     J = J_1l+J_2s -J3
#     J[range(d),range(d)]=0

#     if np.isnan(J).sum()!=0:
#         print('error in J')
        
#     # -------------------------------- find h
#     h1 = 1/2*np.log((1+mu)/(1-mu))

#     h2 = np.zeros(d)
#     for l in trange(d):
#         t2=0
#         for j in range(d):
#             t2+=J[l,j]*mu[j]
#         h2[l]=t2

#     h3 = np.zeros(d)
#     for l in trange(d):
#         t3=0
#         for j in range(d):
#             if j!=l:
#                 t3+= K[l,j]**2 * mu[l]*L[j]
#         h3[l]=t3   

#     h = h1-h2+h3    
    
#     return h,J

# # ----------------------------------- HOPFIELD Monasson
# def hopfield_hj(mu,c, p1,p2):
#     d = len(mu)
#     c_cc = c/np.sqrt((1-mu**2)[:,None]*(1-mu**2)[None,:]) 
#     spectrum, eigvects = np.linalg.eig(c_cc)
#     eigvects = eigvects[np.argsort(spectrum)[::-1]]
#     spectrum = spectrum[np.argsort(spectrum)[::-1]]
#     plt.plot(spectrum,'o')
#     plt.plot((range(len(spectrum)))[:p1], spectrum[:p1],'o', color='red')
#     plt.plot((range(len(spectrum)))[-p2:],spectrum[-p2:],'o', color='orange')
#     plt.axhline(1)
#     print(f'eigenvalues larger than 1 : {(spectrum>1).sum()}')
#     print(f'eigenvalues smaller than 1: {(spectrum<1).sum()}')
#     plt.xlabel('i')
#     plt.ylabel(r'$\lambda_i$')
#     xi1 = (np.sqrt(d*(1-1/spectrum[:p1]))[:,None]*eigvects[:p1]/np.sqrt(1-mu**2))
#     xi2 = (np.sqrt(d*(-1+1/spectrum[-p2:]))[:,None]*eigvects[-p2:]/np.sqrt(1-mu**2))
#     J_h = (xi1[:,None,:]*xi1[:,:,None]).sum(0)/d -  (xi2[:,None,:]*xi2[:,:,None]).sum(0)/d 
#     J_h[range(d),range(d)]=0
#     h_h = np.arctanh(mu) - np.dot(J_h,mu)
#     return h_h, J_h, xi1,xi2
# ___________________________________________________________________________________________


def LOAD_DATA():
    ## _____________ metadata _____________ 
    metadata = pd.read_csv('../../../../scratch/network/csarra/cell_metadata.csv')
    ## _____________ name of cells _____________
    f = h5py.File('../../../../scratch/network/csarra/C57BL6J-638850-raw.h5ad', 'r')
    cell_id = f['obs']['cell_label'][:]
    cell_id = [cell_id[i].decode() for i in range(len(cell_id))]
    brain_section = f['obs']['brain_section_label']['codes'][:]
    ## _____________ manifest _____________
    version = '20231215'
    f = open('../../../../scratch/network/csarra/manifest.json')
    manifest = json.load(f)
    #     print("version: ", manifest['version'])
    ## ---------------------------------------------------
    ##----- (1/2) additional information on cluster
    file = '../../../../scratch/network/csarra/cluster_to_cluster_annotation_membership_pivoted.csv'
    cluster_details = pd.read_csv(file,keep_default_na=False)
    cluster_details.set_index('cluster_alias', inplace=True)
    ##----- (2/2) additional information on colors
    file = '../../../../scratch/network/csarra/cluster_to_cluster_annotation_membership_color.csv'
    cluster_colors = pd.read_csv(file,keep_default_na=False)
    cluster_colors.set_index('cluster_alias', inplace=True)
    metadata_extended = metadata.join(cluster_details,on='cluster_alias')
    metadata_extended = metadata_extended.join(cluster_colors,on='cluster_alias')
    heights = np.sort(np.array(list(dict.fromkeys(metadata_extended.z))))
    ## ---------------------------------------------------
    ## ________________________________________________________
    ## ----- genes
    file = '../../../../scratch/network/csarra/gene.csv.1'
    gene = pd.read_csv(file)
    gene.set_index('gene_identifier',inplace=True)
    print("Number of genes = ", len(gene))
    ## ---- annotadet dataset
    adata = ad.read_h5ad('../../../../scratch/network/csarra/C57BL6J-638850-raw.h5ad', backed='r')
    adatalog = ad.read_h5ad('../../../../scratch/network/csarra/C57BL6J-638850-log2.h5ad', backed='r')
    ## _____________________________________
    print('------\nLOADING in progress: data successfully loaded\n-----')
    # SELECT THE REGION OF THE BRAIN
    section_names  = list(dict.fromkeys(metadata.brain_section_label.values))
    pred = [x in section_names for x in metadata_extended['brain_section_label']]
    sections = metadata_extended[pred] # this is (num_cells)x21
    print(np.array(pred).sum())
    #-------------------
    gnames = gene['gene_symbol'].values[:100]
    pred = [x in gnames for x in adata.var.gene_symbol]
    gene_filtered = adata.var[pred]
    asubset = adata[:,gene_filtered.index].to_memory() #this is 4mln x (num_genes)
    asubsetlog = adatalog[:,gene_filtered.index].to_memory() #this is 4mln x (num_genes)
    #----------------------
    gdata = asubset.to_df()
    gdatalog = asubsetlog.to_df()
    gdata.columns = gnames
    gdatalog.columns = gnames

    joined = sections.join(gdata, 'cell_label')
    joinedlog = sections.join(gdatalog, 'cell_label')
    # -------------------
    data = joined[gnames].to_numpy()
    datalog = joinedlog[gnames].to_numpy()

    print('----------\nLOADING in progress: DATA SUCCESSFULLY SELECTED\n----------')
    # volumes = np.nanmean(data/(2**datalog-1),1) ### only for fraction
    del data
    del datalog
    # ____________________________________________________________________________
    # SELECT THE REGION OF THE BRAIN
    # section_names  = list(dict.fromkeys(metadata.brain_section_label.values))[56:58]
    # select all the regions
    section_names  = list(dict.fromkeys(metadata.brain_section_label.values))
    pred = [x in section_names for x in metadata_extended['brain_section_label']]
    # pred = [x<10 for x in range(len(metadata_extended))]
    sections = metadata_extended[pred] # this is (num_cells)x21
    #     print(np.array(pred).sum())
    #-------------------
    gnames = gene['gene_symbol'].values[:510]
    pred = [x in gnames for x in adata.var.gene_symbol]
    gene_filtered = adata.var[pred]
    asubset = adata[:,gene_filtered.index].to_memory() #this is 4mln x (num_genes)
    #----------------------
    gdata = asubset.to_df()
    gdata.columns = gnames
    joined = sections.join(gdata, 'cell_label')
    # -------------------
    data = joined[gnames].to_numpy()
    print('----------\nLOADING in progress: data selected\n----------')
    # dataf = data/volumes[:,None] ## only for fraction
    classes = sections['class'].values
    listclasses = list(dict.fromkeys(classes))
    NUM_CLASSES = 8
    # ------------------------------------------------------------------
    folder= '../../../../scratch/network/csarra/allen_saved/binary/'
    labels = np.loadtxt(f'{folder}labels.dat') #all dataset label
    classes = labels
    listclasses = np.sort(list(dict.fromkeys(labels)))
    sizes = np.array([(labels==i).sum() for i in listclasses])
    xi = np.vstack([data[classes==listclasses[i]][:,:500] for i in np.argsort(sizes)[::-1][:NUM_CLASSES]])
    ordered_listclasses = listclasses[np.argsort(sizes)[::-1]]
    label = np.hstack([[ordered_listclasses[i]]*np.sort(sizes)[::-1][i] for i in range(NUM_CLASSES)])
    label_list = np.array(list(dict.fromkeys(label))).astype(int)
    labels=label
    data_classes = np.vstack(xi)
    print('DONE - Loading completed') 
    return data_classes, labels