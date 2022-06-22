import numpy as np
from itertools import compress
import random
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
plt.rcParams['figure.figsize'] = (16, 20)
from mpl_toolkits.mplot3d import Axes3D 
import tikzplotlib
from itertools import chain, combinations
import pickle
import sys
import sklearn.linear_model as sklm
import quadprog
import functools
import time
#matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})
float_formatter = "{:.3e}".format
#np.set_printoptions(formatter={'float_kind':float_formatter})
#plt.rcParams.update({'font.size': 22})

def timer(func):
    #If the boolean in the if is turned to True, this will print the running time of every function that is decorated by @timer with its name and let us see what takes how much time
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        if False:
            print("Finished {} in {} secs".format(repr(func.__name__), round(run_time, 3)))
        return value

    return wrapper

@timer
def get_alives(x,previous=None,thresh=1e-12,debug=False,flag_issue=False):
    #This function returns a list of boolean telling us for each vector whether it is alive or not. If it receives the previous list and flag_issue is false, it only updates elements that were previously alive
    n=len(x)
    ones=np.ones(n)
    if debug:
        print(f'Computing the list of elements that are alive with threshold {thresh}')
        print(f'x in basis: {x}')
    if flag_issue:
        if sum((x+ones)>-thresh)<n or sum((ones-x)>-thresh)<n:
            x[((x+ones)<-thresh) | ((ones-x)<-thresh)]=np.round(x[((x+ones)<-thresh) | ((ones-x)<-thresh)])
            if sum((x+ones)>-thresh)<len(x) or sum((ones-x)>-thresh)<len(x):
                print('Issue: x is out of bound!')
                debug=True
    if previous is None:
        alive=(np.abs(x+ones)>thresh) & (np.abs(x-ones)>thresh)
    else:#is this even faster with all the indexing ? hard to know
        alive_update=(np.abs(x[previous]+ones[previous])>thresh) & (np.abs(x[previous]-ones[previous])>thresh)
        alive=previous.copy()
        alive[previous]=alive_update
    return alive,debug,None if previous is None else [i for i in range(len(alive)) if ((previous[i]) and not alive[i])]

@timer
def choose_pivot(v,x,alive,mode='random',debug=False):
    if debug:
        print(f'Choosing pivot through mode {mode}.')
        print(f'x in basis: {x}')
        print(f'Alive: {alive}')
    if mode=='max_index':
        for i in range(len(x),0,-1):
            if alive[i]:
                return i
        pivot= -1
    elif mode=='random':
        try:
            pivot= random.choice(np.arange(len(x))[alive])
        except IndexError:
            #print('Every element is fixed')
            pivot= -1
    elif mode=='max_norm':
        norms=[norm(v[i]) if alive[i] else 0 for i in range(len(v))]
        if debug:
            print(f'norms: {norms}')
        pivot= np.argmax(norms) if max(norms)!=0 else -1
    elif mode=='norm':
        norms=[norm(v[i]) if alive[i] else 0 for i in range(len(v))]
        if max(norms)!=0:
            r=random.random()*sum(norms)
            cumsum_norms=np.cumsum(norms)
            cumsum_norms[cumsum_norms-r<0]=float('Inf')
            pivot=np.argmin(cumsum_norms)
        else:
            pivot=-1
    elif mode=='coloring':
        if sum(alive)==0:
            pivot=-1
        elif max(np.abs(x[alive]))==0:
            pivot=random.choice(np.arange(len(x))[alive])
        else:
            coloring_values=np.abs(x)
            coloring_values[~alive]=0
            r=random.random()*sum(coloring_values)
            cumsum_col=np.cumsum(coloring_values)
            cumsum_col[cumsum_col-r<0]=float('Inf')
            pivot=np.argmin(cumsum_col)
    elif mode=='max_coloring':
        if sum(alive)==0:
            pivot=-1
        elif max(np.abs(x[alive]))==0:
            pivot=random.choice(np.arange(len(x))[alive])
        else:
            coloring_values=np.abs(x)
            coloring_values[~alive]=0
            pivot=np.argmax(coloring_values)
    else:
        print('Unknown mode of pivot choice: aborting.')
        pivot=None
    if debug:
        print(f'Pivot chosen: {pivot}')
    return pivot

@timer
def next_direction(p,v,a,b,B,alive,old_alive_and_not_pivot,old_alive,X_t=None,debug=False,bigger_first=False,force_balance=False,fast_lst_sq=True,solve_adversarial=False,d_instead_of_d_inv=False,i_instead_of_d_inv=False,no_matrix_mult=True,flag_issue=False,B_S=None,C_S=None,return_all=False):
    alive_count=sum(alive)
    n=len(v)
    u=np.zeros(n)
    u[p]=1
    alive_and_not_pivot=alive.copy()
    alive_and_not_pivot[p]=False
    if debug:
        print(f'Alive and not pivot: {alive_and_not_pivot}')
    alive_and_not_pivot=np.asarray(alive_and_not_pivot)
    B_t=B[:,alive_and_not_pivot]
    if alive_and_not_pivot.any() and (((bigger_first or solve_adversarial) and not no_matrix_mult) or flag_issue or debug):
        q,r=np.linalg.qr(B[:,alive_and_not_pivot])
        rs=np.array([(r[i,:]==np.zeros(r.shape[1])).all() for i in range(r.shape[0])])
        if len(rs)!=0:
            q=q[:,~rs]
            v_perp=B[:,p]-q.dot(np.linalg.inv(q.T.dot(q))).dot(q.T).dot(B[:,p])
        else:
            v_perp=B[:,p]
    else:
        v_perp=B[:,p]
    if ((v_perp)<1e-12).all() and bigger_first:
        model=sklm.Lasso(fit_intercept=False,alpha=1e-32)
        model.fit(B_t,-B[:,p])
        u1=model.coef_
        colinear=True
        u[alive_and_not_pivot]=u1
    elif not force_balance:
        all_good=False
        if solve_adversarial and ((v_perp)<1e-12).all():
            B_min=np.vstack((B,np.ones(n)))
            P=B_min.T.dot(B_min)+1e-15*np.eye(B_min.shape[1])
            A=np.eye(n)[~alive_and_not_pivot,:]
            #A=np.vstack((A,np.ones(len(v))))
            b=np.zeros(n)
            b[p]=1
            b=b[~alive_and_not_pivot]
            #b=np.append(b,0)
            u=quadprog_solve_qp(P,np.zeros(n),A=A,b=b)
            colinear=False
            all_good=(v_perp-B.dot(u)<1e-12).all()
        if not all_good and solve_adversarial and ((v_perp)<1e-12).all():
            u=np.zeros(n)
            u[p]=1
            print('Additional constraints breached the framework and were thus discarded')
        if not all_good:
            if fast_lst_sq and B_t.shape[0]>=B_t.shape[1] and not no_matrix_mult:
                if X_t is None:
                    X_t=inv(B_t.T.dot(B_t))
                else:
                    indices_to_update=[sum(alive_and_not_pivot[:x]) for x in range(alive_and_not_pivot.shape[0]) if ((not alive_and_not_pivot[x]) and old_alive_and_not_pivot[x])]
                    indices_to_update.reverse()
                    if debug:
                        print(f'indices to update: {indices_to_update}')
                    t_before_update=time.perf_counter()
                    try:
                        for k in indices_to_update:
                            length=X_t.shape[0]
                            X_t_k=X_t[:,k]
                            update=X_t[k,k]**-1*np.matmul(X_t_k.reshape((length,1)),X_t_k.reshape((1,length)))
                            if debug:
                                print(f'X_t:{X_t}')
                                print(f'update: {update}')
                            X_t-=X_t-update
                            X_t=np.delete(X_t,k,0)
                            X_t=np.delete(X_t,k,1)
                    except IndexError:
                        print(f'x:{x}')
                    if debug:
                        print(f'Time necessary to update X_t: {time.perf_counter()-t_before_update}')
                        error=X_t-inv(B_t.T.dot(B_t))
                        if error.shape[0]!=0 and np.max(np.abs(error))>1e-9:
                            print(f'X_t:{X_t}')
                            print(f'error:{error}')
                            print(np.max(np.abs(error)))
                #if (X_t!=np.array([0])).all():
                t_before_update=time.perf_counter()
                u1=np.matmul(np.matmul(X_t,(B_t.T)),(-B[:,p]))
                #else:
                #    print('Replacing X_t by zero')
                #    u1=np.zeros(1)
                if debug:
                    print(f'Time necessary to compute u_1 from X_t: {time.perf_counter()-t_before_update}')
                    u1_=np.linalg.lstsq(B_t,-B[:,p])[0]
                    error=u1-u1_
                    if error.shape[0]!=0 and np.max(np.abs(error))>1e-9:
                        print(f'final error:{error}')
                        print(np.max(np.abs(error)))
            elif no_matrix_mult and n<=len(v[0]):#matrix names correspond to the originals in gsw_notes.pdf
                S=alive
                #if A_S is None:
                #    A_S=B.copy()
                #    if debug:
                #        print(f'A_S: {A_S}')
                #I_S=np.diag(np.array([1 if S[i] else 0 for i in range(len(S))]))
                if C_S is None or flag_issue:
                    #A_S=B.copy()
                    if debug:
                        print(f'A_S:{B}')

                    if C_S is None:
                        C_S=inv(B)
                    #C_S2=np.linalg.inv(A_S.dot(A_S.T)).dot(A_S).T
                    if debug:
                        print(f'C_S:{C_S}')
                        #print(f'C_S2:{C_S2}')
                        print(f'C_S*A_S:{C_S.dot(B)}')
                if B_S is None:
                    B_S=np.matmul(C_S,C_S.T)
                    if debug:
                        print(f'B_S: {B_S}')
                t_before_update=time.perf_counter()
                indices_to_update=[i for i in range(n) if (not alive[i]) and (old_alive[i])]
                indices_to_update.reverse()
                if debug:
                    print(f'indices to update: {indices_to_update}')
                try:
                    for k in indices_to_update:
                        b_k=B_S[:,k]
                        update_B=b_k[k]**-1*np.matmul(b_k.reshape((n,1)),b_k.reshape((1,n)))
                        update_C=b_k[k]**-1*np.matmul(b_k.reshape((n,1)),C_S[k,:].reshape((1,C_S.shape[1])))
                        if debug:
                            print(f'B_S:{B_S}')
                            print(f'update of B_S: {update_B}')
                            print(f'C_S:{C_S}')
                            print(f'update of C_S: {update_C}')
                        B_S-=update_B
                        C_S[alive]-=update_C[alive]
                        #A_S=B.copy()
                        #A_S[:,~alive]=0
                        #print(C_S-np.linalg.inv(A_S.T.dot(A_S)).dot(A.T))
                except IndexError:
                    print('Error in updating in no_matrix_mult part')
                    print(f'x:{x}')
                if return_all:
                    u_2=[]
                    for i in range(len(alive)):
                        if alive[i]:
                            update=B_S[i,:]/B_S[i,i]
                            v_perp_2=C_S[i,:]/norm(C_S[i,:])**2
                            to_send=(i,update,v_perp_2)
                            #print(to_send)
                            u_2.append(to_send)
                else:
                    u_2=B_S[p,:]/B_S[p,p]
                #u_2[~alive]=0
                if flag_issue or debug:
                    v_perp_2=C_S[p,:]/norm(C_S[p,:])**2
                    #A_S=B.copy()
                    #A_S[:,~alive]=0
                    #print(C_S[p,:].reshape((1,C_S.shape[1])).T-np.matmul(A_S.dot(C_S),C_S[p,:].reshape((1,C_S.shape[1])).T))
                if debug:
                    print(f'Time necessary to update C_S and D_S: {time.perf_counter()-t_before_update}')
                if debug or (flag_issue and max(np.abs(v_perp_2-v_perp))>1e-10):
                    print(f'Error in v_perp:{v_perp_2-v_perp}')
            else:
                u1=np.linalg.lstsq(B_t,-B[:,p])[0]
            if flag_issue and no_matrix_mult:
                u1=np.linalg.lstsq(B_t,-B[:,p])[0]
            if d_instead_of_d_inv:
                d=np.diag([norm(B_t[:,i]) for i in range(B_t.shape[1])])
                u1=d.dot(d).dot(u1)
            if i_instead_of_d_inv:
                d=np.diag([norm(B_t[:,i]) for i in range(B_t.shape[1])])
                u1=d.dot(u1)
            colinear=False
            if not no_matrix_mult or flag_issue or n>len(v[0]):
                u[alive_and_not_pivot]=u1
                if (no_matrix_mult and n<=len(v[0])) and (debug or (flag_issue and max(np.abs(u_2-u))>1e-10)):
                    print(f'classic u:{u}')
                    print(f'new u:{u_2}')
                    print(f'Error in u:{u_2-u}')
            else:
                u=u_2
        #if no_matrix_mult and (debug or (flag_issue and max(np.abs(u_2-u))>1e-10)):
        #    print(f'classic u:{u}')
        #    print(f'new u:{u_2}')
        #    print(f'Error in u:{u_2-u}')
    else:
        P=B.T.dot(B)
        A=np.eye(n)[~alive_and_not_pivot,:]
        A=np.vstack((A,np.ones(n)))
        b=np.zeros(n)
        b[p]=1
        b=b[~alive_and_not_pivot]
        b=np.append(b,0)
        u=quadprog_solve_qp(P,np.zeros(n),A=A,b=b)
        colinear=False
    if (debug or (flag_issue and max(np.abs(v_perp-B.dot(u)))>1e-6 and not force_balance and not i_instead_of_d_inv and not d_instead_of_d_inv)) and not return_all:
        #print('Issue with classic u')
        print(f'v_perp:{v_perp}')
        print(f'v_perp-sum u_i*v_i:{v_perp-B.dot(u)}')
    if (no_matrix_mult and n<=len(v[0]) and (debug or (flag_issue and max(np.abs(v_perp-B.dot(u_2)))>1e-6 and not force_balance and not i_instead_of_d_inv and not d_instead_of_d_inv))) and not return_all:
        #print('Issue with u from no_mat_mult')
        print(f'v_perp:{v_perp}')
        print(f'v_perp-sum u_i*v_i:{v_perp_2-B.dot(u_2)}')
    if debug:
        print(f'Calculated update direction u:{u}')
    return u,colinear,X_t,alive_and_not_pivot,alive,B_S,C_S

def quadprog_solve_qp(P, q, A=None, b=None):
    qp_G = .5 * (P + P.T)   # make sure P is symmetric
    qp_a = -q
    qp_C = -A.T
    qp_b = -b
    meq = A.shape[0]
    return quadprog.solve_qp(qp_G, qp_a, qp_C, qp_b, meq)[0]

@timer
def next_factor(x_in_basis,u,p,a,b,colinear,debug=False,smallest_delta=False,bigger_first=False,thresh=0,mode=None,always_new_pivot=False):#do two cases in return/append instead of doubling code
    return_all_modes=['move_inv_prop','min_move_random']
    if not mode in return_all_modes:        
        non_zero=np.abs(u)>1e-10
        deltas=np.concatenate(((a[non_zero]-x_in_basis[non_zero])/u[non_zero],(b[non_zero]-x_in_basis[non_zero])/u[non_zero]),axis=0)
        if debug:
            print(f'All deltas considered:{deltas}')
        try:
            d_p=min(deltas[deltas>thresh])
            if debug:
                print(f'delta_+:{d_p}')
        except:
            print(f'No delta>=0: {deltas}')#could set d_p=0 maybe
        try:
            d_m=max(deltas[deltas<-thresh])
            if debug:
                print(f'delta_-:{d_m}')
        except:
            print(f'No delta<=0: {deltas}')#could set d_m=0 maybe
        if d_p<1e-12 and d_m>-1e-12:
            print('Issue: too small deltas')
        if not bigger_first or not colinear or abs(x_in_basis[p])<=thresh:
            r=random.random()
            if r>d_p/(d_p-d_m) or (smallest_delta and d_p<abs(d_m)):
                if debug:
                    print('delta=delta_+')
                return d_p,d_m,p,u
            else:
                if debug:
                    print('delta=delta_-')
                return d_m,d_p,p,u
        else:
            if abs(x_in_basis[p])>0:
                if debug:
                    print('delta=delta_+')
                return d_p,d_m,p,u
            else:
                if debug:
                    print('delta=delta_-')
                return d_m,d_p,p,u
    else:
        deltas_array=[]
        for i in range(len(u)):
            (p,u_,v_perp)=u[i]
            #print(u_)
            non_zero=np.abs(u_)>1e-10
            #print(non_zero)
            deltas=np.concatenate(((a[non_zero]-x_in_basis[non_zero])/u_[non_zero],(b[non_zero]-x_in_basis[non_zero])/u_[non_zero]),axis=0)
            if debug:
                print(f'All deltas considered:{deltas}')
            try:
                d_p=min(deltas[deltas>thresh])
                if debug:
                    print(f'delta_+:{d_p}')
            except:
                print(f'No delta>=0: {deltas}')#could set d_p=0 maybe
            try:
                d_m=max(deltas[deltas<-thresh])
                if debug:
                    print(f'delta_-:{d_m}')
            except:
                print(f'No delta<=0: {deltas}')#could set d_m=0 maybe
            if d_p<1e-12 and d_m>-1e-12:
                print('Issue: too small deltas')
            if not bigger_first or not colinear or abs(x_in_basis[p])<=thresh:
                r=random.random()
                if r>d_p/(d_p-d_m) or (smallest_delta and d_p<abs(d_m)):
                    if debug:
                        print('delta=delta_+')
                    deltas_array.append((d_p,d_m))
                else:
                    if debug:
                        print('delta=delta_-')
                    deltas_array.append((d_m,d_p))
        norms=[(norm(deltas_array[i][0]*u[i][2]),norm(deltas_array[i][1]*u[i][2])) for i in range(len(u))]
        if debug:
            print(f'Potential norms of move: {norms}')
        if mode=='min_move_random':
            if len(norms)==0:
                return 0,0,-1,np.zeros(len(x_in_basis))
            elif smallest_delta==False:
                i=np.argmin([norms[i][0] for i in range(len(norms))])
                return deltas_array[i][0],deltas_array[i][1],u[i][0],u[i][1]
            else:
                i=np.argmin([min(norms[i][0],norms[i][1]) for i in range(len(norms))])
                j=np.argmin(norms[i])
                return deltas_array[i][j],deltas_array[i][(j-1)%2],u[i][0],u[i][1]
        elif mode=='move_inv_prop':
            norms=[2*norm(deltas_array[i][0]*u[i][2]*deltas_array[i][1]/(abs(deltas_array[i][0])+abs(deltas_array[i][1]))) for i in range(len(u))]
            if len(norms)==0:
                return 0,0,-1,np.zeros(len(x_in_basis))
            else:
                norms_zero=[n==0 for n in norms]
                if sum(norms_zero)==0:
                    norms_inv=[1/n for n in norms]
                    r=random.random()*sum(norms_inv)
                    norms_cs=np.cumsum(norms_inv)
                    norms_cs[norms_cs-r<0]=float('Inf')
                    pivot=np.argmin(norms_cs)
                    r=random.random()
                    #print('d_m then d_p')
                    d_m=min(deltas_array[pivot])
                    #print(d_m)
                    d_p=max(deltas_array[pivot])
                    #print(d_p)
                    #print(u[pivot][0],pivot)
                    #print(u[pivot][1])
                    if r>d_p/(d_p-d_m) or (smallest_delta and d_p<abs(d_m)):
                        if debug:
                            print('delta=delta_+')
                        return d_p,d_m,u[pivot][0],u[pivot][1]
                    else:
                        if debug:
                            print('delta=delta_-')
                        return d_m,d_p,u[pivot][0],u[pivot][1]
                else:
                    print('This case shouldnt arise as no_matrix_mult requires C_S to actually exist and it doesnt if v_perp is 0 , error somewhere')
        else:
            print('unknown mode in next_factor')
            return None
        

@timer
def change_basis(v,basis1,basis2):#from basis1 to basis2
    return np.matmul(np.linalg.inv(np.transpose(np.array(basis2))),np.matmul(np.transpose(np.array(basis1)),v))

@timer
def orthonormal_basis(n):
    basis=[]
    for i in range(n):
        v_i=np.zeros(n)
        v_i[i]=1
        basis.append(v_i)
    return basis

@timer
def gram_schmidt_walk(v,x,a=None,b=None,plot=False,debug=False,smallest_delta=False,basis=None,order=False,bigger_first=False,force_balance=False,fast_lst_sq=True,return_pivot_in_colored=False,mode=None,return_pivots=False,pivot=None,d_instead_of_d_inv=False,i_instead_of_d_inv=False,no_matrix_mult=True,flag_issue=False,early_stop=None,always_new_pivot=False):
    n=len(x)
    return_all_modes=['move_inv_prop','min_move_random']
    orth_basis=orthonormal_basis(n)
    if a is None:
        if debug:
            print('Initializing a with -1s')
        a=-np.ones(n)
    if b is None:
        if debug:
            print('Initializing b with 1s')
        b=np.ones(n)
    if basis is not None and len(basis)<n:
        print('Basis is lacking vectors to be full-dimensional: replacing it by a canonical orthonormal basis')#could complete it with Gram-Schmidt maybe
        basis=None
    if basis is not None and np.linalg.cond(np.array(basis)) > 1/sys.float_info.epsilon:
        print('Basis matrix is singular: replacing it by a canonical orthonormal basis')
        basis=None
    if sum(a<b)<n:
        print('Issue with hyper parallelepipeds: a>b for some dimension')
    alive,debug,_=get_alives(x,debug=debug,flag_issue=flag_issue)
    if not (mode in return_all_modes and no_matrix_mult==True and n<=len(v[0])):
        if pivot is None:
            p=choose_pivot(v,x,alive,debug=debug,mode=mode if mode is not None else'random' if not bigger_first else 'max_norm')
        else:
            p=pivot
    else:
        p=-2
    if basis is None:
        x_in_basis=x.copy()
    else:
        x_in_basis=change_basis(x.copy(),orth_basis,basis)
    i=0
    colored=[]
    X_t=None
    B_S,C_S=None,None
    old_alive_and_not_pivot=alive
    old_alive=alive
    pivot_in_colored=0 
    pivots=[]
    B=np.transpose(np.vstack(tuple([e for e in v])))
    if n>len(v[0]) and no_matrix_mult:
        B=np.concatenate((B,1e-10*np.eye(n)),axis=0)
    if basis is not None:
        B=np.matmul(B,np.vstack(tuple([e for e in basis])).T)#is v already a list ? If so we can simplify syntax here
    while p!=-1:
        if debug:
            print(f'\n Iteration {i}')
        if early_stop is not None and i==early_stop:
            if debug:
                print(f'Stopping early after step {i-1}')
            break
        u_in_basis,colinear,X_t,old_alive_and_not_pivot,old_alive,B_S,C_S=next_direction(p,v,a,b,B,alive,old_alive_and_not_pivot,old_alive,X_t,debug=debug,bigger_first=bigger_first,force_balance=force_balance,fast_lst_sq=fast_lst_sq,i_instead_of_d_inv=i_instead_of_d_inv,d_instead_of_d_inv=d_instead_of_d_inv,B_S=B_S,C_S=C_S,no_matrix_mult=no_matrix_mult,flag_issue=flag_issue,return_all=mode in return_all_modes)
        d1,d2,p,u_in_basis=next_factor(x_in_basis,u_in_basis if (i==0 or (not (mode in return_all_modes)) or always_new_pivot or not alive[p]) else [u[1] for u in u_in_basis if u[0]==p][0],p,a,b,colinear,debug=debug,smallest_delta=smallest_delta,bigger_first=bigger_first,mode=None if (not always_new_pivot and alive[p] and i!=0) else mode,always_new_pivot=always_new_pivot)
        pivots.append(p)

        if basis is not None:
            u=change_basis(u_in_basis,basis,orth_basis)
        else:
            u=u_in_basis
        if plot:
            plot_situation_v2(v,p,x,u,[d1,d2],i)
        x_in_basis+=d1*u_in_basis
        x+=d1*u
        if debug:
            print(f'Incurred discrepancy:{norm(np.matmul(np.transpose(np.vstack(tuple([e for e in v]))),x))}')
        alive,debug,newly_colored=get_alives(x_in_basis,previous=alive,debug=debug,flag_issue=flag_issue)
        if debug:
            print('')
        colored.extend(newly_colored)
        if p in newly_colored:
            pivot_in_colored+=1
        if not alive[p] and not (mode in return_all_modes and no_matrix_mult==True and n<=len(v[0])):
            p=choose_pivot(v,x_in_basis,alive,debug=debug,mode=mode if mode is not None else'random' if not bigger_first else 'max_norm')
        x_in_basis[~alive]=np.round(x_in_basis[~alive])
        i+=1
        if i-5>n:
            print('Issue, the algorithm took more steps than expected')
            print(f'x:{x}')
            print(f'u:{u}')
            print(f'delta: {d1}')
            print(f'Iteration {i}')
            return None
        if debug:
            print('in basis')
            print(f'x:{x_in_basis}')
            print(f'u:{u_in_basis}')
    if order:
        return x,colored
    if return_pivots:
        return x,pivots
    elif return_pivot_in_colored:
        return x,pivot_in_colored/i
    else:
        return x

def gram_schmidt_columns(X):
    Q, R = np.linalg.qr(X)
    return Q

def sample_from_sphere(n):
    thetas=np.random.uniform(0,2*np.pi,n)
    return [np.array([np.sin(t),np.cos(t)]) for t in thetas]

@timer
def sample_from_ball(n,d=2):
    p=[]
    for i in range(n):
        u = np.random.normal(0,1,d)  # an array of d normally distributed random variables
        norm=np.sum(u**2) **(0.5)
        r = random.random()**(1.0/d)
        p.append(r*u/norm)
    return p
    #r = np.random.uniform(0,1,n)**0.5
    #theta = np.random.uniform(0,2*np.pi,n)
    #x = r*np.cos(theta)
    #y = r*np.sin(theta)
    #return [np.array([x[i],y[i]]) for i in range(n)]
    
def sample_binary(n,d,p=0.5):
    return [np.random.binomial(size=d, n=1, p=p) for i in range(n)]

def inv(m):
    a,_ = m.shape
    i = np.eye(a, a)
    sol=np.linalg.lstsq(m, i)
    #print(sol[3])
    return sol[0]

def latex_vector(x):
    return str(list(x)).replace('[','').replace(']','').replace(',','\\ \n')

def plot_situation(v,p,x,u,deltas,i):
    #plot vectors, combination where it is now, two potential updates,show x and deltas and u
    plt.plot([e[0] for e in v],[e[1] for e in v],'o',label='Input vector')
    B=np.transpose(np.vstack(tuple([e for e in v])))
    plt.plot(np.matmul(B,x)[0],np.matmul(B,x)[1],'X',label='Current relaxation',markersize=15)
    x_1=x+deltas[0]*u
    x_2=x+deltas[1]*u
    xs=[x_1,x_2]
    print(f'xs: {xs}')
    print(f'sum:{[np.matmul(B,x) for x in xs]}')
    plt.plot([np.matmul(B,x)[0] for x in xs],[np.matmul(B,x)[1] for x in xs],'H',label='Potential updated relaxation')
    #plt.figtext(0.63,0.05, f"$x_t=\begin%{latex_vector(np.round(x,3))}\end%$\\ \n $u_t=\begin%{latex_vector(np.round(u,3))}\end%$\\ \n $\delta^+_t,\delta^-_t ={str(list(np.round(np.array(deltas),3))).replace('[','').replace(']','')}$".replace('%','{pmatrix}'))
    plt.figtext(0.63,0.05, f"$\delta^+_t,\delta^-_t ={str(list(np.round(np.array(deltas),3))).replace('[','').replace(']','')}$")
    plt.legend(bbox_to_anchor=(1.8, 1))
    tikzplotlib.save(f"gswalk{i}2d.tex")
    plt.show()
    if x.shape[0]==3:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.gca(projection='3d')

        sn = 1   #limits in x,y,z
        x1, x2 = -sn, sn
        y1, y2 = -sn, sn    
        z1, z2 = -sn, sn
        ax.scatter(x[0],x[1],x[2],label='Current coloring')
        ax.scatter([x_1[0],x_2[0]], [x_1[1],x_2[1]],zs=[x_1[2],x_2[2]])

        # Data for plotting plane x|y|z=0 within the domain
        tmp = np.linspace(-1, sn, 8)
        x,y = np.meshgrid(tmp,tmp)
        z = 0*x

        ax.plot_surface(z+1,x,y, alpha=0.15, color='red')    # plot the plane x=1
        ax.plot_surface(z-1,x,y, alpha=0.15, color='red')    # plot the plane x=-1
        ax.plot_surface(x,z+1,y, alpha=0.15, color='green')  # plot the plane y=1
        ax.plot_surface(x,z-1,y, alpha=0.15, color='green')  # plot the plane y=-1
        ax.plot_surface(x,y,z+1, alpha=0.15, color='blue')   # plot the plane z=1
        ax.plot_surface(x,y,z-1, alpha=0.15, color='blue')   # plot the plane z=-1
        ax.plot([x_1[0],x_2[0]], [x_1[1],x_2[1]],zs=[x_1[2],x_2[2]],label='Update direction')

        # Set limits of the 3D display
        ax.set_xlim3d([-sn, sn])
        ax.set_ylim3d([-sn, sn])
        ax.set_zlim3d([-sn, sn])

        # Set labels at the 3d box/frame
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.savefig(f"gswalk{i}3d.png")
        plt.legend()
        tikzplotlib.save(f"gswalk{i}3d.tex")
        plt.show()
        
def plot_situation_v2(v,p,x,u,deltas,i):
    #fig, (ax1, ax2) = plt.subplots(1, 2)
    fig = plt.figure(figsize=plt.figaspect(0.5))
    fig.subplots_adjust(wspace=0.1, hspace=0)
    ax1 = fig.add_subplot(1, 2, 2)
    #plot vectors, combination where it is now, two potential updates,show x and deltas and u
    ax1.plot([v[e][0] for e in range(len(v)) if np.abs(x[e])<1],[v[e][1] for e in range(len(v)) if np.abs(x[e])<1],'o',label='Colorless vector')
    ax1.plot([v[e][0] for e in range(len(v)) if np.abs(x[e])==1],[v[e][1] for e in range(len(v)) if np.abs(x[e])==1],'o',label='Colored vector')
    ax1.plot(v[p][0],v[p][1],'*',label='Pivot vector',markersize=12)
    B=np.transpose(np.vstack(tuple([e for e in v])))
    ax1.plot(np.matmul(B,x)[0],np.matmul(B,x)[1],'X',label='Current balance',markersize=15)
    x_1=x+deltas[0]*u
    x_2=x+deltas[1]*u
    xs=[x_1,x_2]
    print(f'xs: {xs}')
    print(f'sum:{[np.matmul(B,x) for x in xs]}')
    ax1.plot([np.matmul(B,x)[0] for x in xs],[np.matmul(B,x)[1] for x in xs],'<',label='Potential update')
    #plt.figtext(0.63,0.05, f"$x_t=\begin%{latex_vector(np.round(x,3))}\end%$\\ \n $u_t=\begin%{latex_vector(np.round(u,3))}\end%$\\ \n $\delta^+_t,\delta^-_t ={str(list(np.round(np.array(deltas),3))).replace('[','').replace(']','')}$".replace('%','{pmatrix}'))
    plt.figtext(0.45,0.04, f"$\delta^+_t,\delta^-_t ={str(list(np.round(np.array(deltas),3))).replace('[','').replace(']','')}$")
    #ax1.legend(bbox_to_anchor=(0,0),loc='lower center')

    #tikzplotlib.save(f"gswalk{i}2d.tex")
    #plt.show()
    if x.shape[0]==3:
        #fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 2, 1, projection='3d')

        sn = 1   #limits in x,y,z
        x1, x2 = -sn, sn
        y1, y2 = -sn, sn    
        z1, z2 = -sn, sn

        # Data for plotting plane x|y|z=0 within the domain
        tmp = np.linspace(-1, sn, 8)
        x_,y = np.meshgrid(tmp,tmp)
        z = 0*x_

        ax.plot_surface(z+1,x_,y, alpha=0.15, color='red')    # plot the plane x=1
        ax.plot_surface(z-1,x_,y, alpha=0.15, color='red')    # plot the plane x=-1
        ax.plot_surface(x_,z+1,y, alpha=0.15, color='green')  # plot the plane y=1
        ax.plot_surface(x_,z-1,y, alpha=0.15, color='green')  # plot the plane y=-1
        ax.plot_surface(x_,y,z+1, alpha=0.15, color='blue')   # plot the plane z=1
        ax.plot_surface(x_,y,z-1, alpha=0.15, color='blue')   # plot the plane z=-1
        ax.scatter(x[0],x[1],x[2],label='Current coloring')
        ax.scatter([x_1[0],x_2[0]], [x_1[1],x_2[1]],zs=[x_1[2],x_2[2]],label='Potential update')
        ax.plot([x_1[0],x_2[0]], [x_1[1],x_2[1]],zs=[x_1[2],x_2[2]],label='Update direction')

        # Set limits of the 3D display
        ax.set_xlim3d([-sn, sn])
        ax.set_ylim3d([-sn, sn])
        ax.set_zlim3d([-sn, sn])

        # Set labels at the 3d box/frame
        #ax.set_xlabel('X')
        #ax.set_ylabel('Y')
        #ax.set_zlabel('Z')
        if sum(np.abs(x)==1)==2:
            ax.legend(bbox_to_anchor=(0.8, -0.07))
    if sum(np.abs(x)==1)==2:
        ax1.legend(bbox_to_anchor=(1.1, -0.07),ncol=2)
    plt.savefig(f"gswalkboth{i}.pdf", bbox_inches='tight')
    tikzplotlib.save(f"gswalk{i}both.tex")
    plt.show()
    
def naive_walk(vs_,minimizing=True):
    vs=vs_.copy()
    indices=list(range(len(vs)))
    random.shuffle(indices)
    x=np.zeros(len(vs))
    output=np.zeros(len(vs[0]))
    for i in indices:
        v=vs[i]
        if (norm(output+v)-norm(output-v))*(1 if minimizing else -1)<0 :
            x[i]=1
            output+=v
        else:
            x[i]=-1
            output-=v
    return x

def norm(v):
    return np.sqrt(sum(v**2))

def normalize(v):
    return v/norm(v)

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def best_coloring(v,minimizing=True):
    indices_subset=powerset(range(len(v)))
    v_mat=np.transpose(np.array(v))
    best_assignment=[]
    best_disc=None
    for i_s in indices_subset:
        assignment=np.array([1 if i in i_s else -1 for i in range(len(v))])
        disc=norm(np.matmul(v_mat,assignment))
        if best_disc is None or (disc-best_disc)*(1 if minimizing else -1)<0:
            best_disc=disc
            best_assignment=assignment
    return best_disc,best_assignment

def open_dic_file(file_name):
    try:
        file = open(file_name, "rb")
        dic = pickle.load(file)
        file.close()
        return dic
    except FileNotFoundError:
        return {}

def save_dic_to_file(dic,file_name):
    file = open(file_name,'wb')
    pickle.dump(dic,file)
    file.close()
    
def average(list_):
    return sum(list_)/len(list_)

def produce_correlations(v,file_name=None,repeat=10**3,thresh1=0.5,thresh2=0.2):
    print(f'Studying correlations with {repeat} iterations')
    n=len(v)
    gsw_xs=[]
    coloring_order=[]
    when_colored=[]
    for i in range(repeat):
        print(f'\n Try #{i}')
        res,order=gram_schmidt_walk(v,np.zeros(n),order=True,smallest_delta=False)
        gsw_xs.append(np.array(res))
        coloring_order.append(order)
        switched_list=[(i,order[i]) for i in range(len(order))]
        switched_list.sort(key=lambda x:x[1])
        when_colored.append((np.array([x[0] for x in switched_list])).astype(float))
    coloring_correlation=np.corrcoef(np.vstack(gsw_xs),rowvar=False)
    order_correlation=np.corrcoef(np.vstack(when_colored),rowvar=False)
    if file_name is not None:
        dic={'v':v,'gsw_xs':gsw_xs,'coloring_order':coloring_order,'when_colored':when_colored,'coloring_correlation':coloring_correlation,'ordering_correlation':ordering_correlation}
        save_dic_to_file(dic,'file_name')
    return coloring_correlation, order_correlation

def study_correlations(coloring_correlation, order_correlation,thresh1=0.5,thresh2=0.2):
    total_sum=0
    min_dist=10**3
    max_dot_prod=0
    for i in range(coloring_correlation.shape[0]):
        for j in range(coloring_correlation.shape[0]):
            if i<j:
                dist=norm(v[i]-v[j])
                if dist<min_dist:
                    min_dist=dist
                dot_prod=v[i].dot(v[j])
                if abs(dot_prod)>abs(max_dot_prod):
                    max_dot_prod=dot_prod
                total_sum+=dist
                if abs(order_correlation[i,j])>thresh2:
                    print('order')
                    print(f'\nvectors {i} and {j}')
                    print(order_correlation[i,j])
                    print(v[i])
                    print(v[j])
                    print(norm(v[i]-v[j]))
                    print(v[i].dot(v[j]))
                if abs(coloring_correlation[i,j])>thresh1:
                    print('coloring')
                    print(f'vectors {i} and {j}')
                    print(coloring_correlation[i,j])
                    print(v[i])
                    print(v[j])
                    print(norm(v[i]-v[j]))
                    print(v[i].dot(v[j]))
    total_sum/=(n*(n-1))/2
    print(f'mean distance between vectors: {total_sum}')
    print(f'min distance: {min_dist}')
    print(f'max dot prod: {max_dot_prod}')