# imports - reduce this to the necessities later
import numpy as np
from sympy import *
import os
import scipy
from scipy.optimize import fsolve, minimize
from copy import deepcopy

# Some useful functions
def add_Pkp1res_stat_dict_vals( dic, Pkp1res_stat, Pkp1res_stat_val, states_kp1res, Nres, Ncorr ):
    for res in range(Nres-Ncorr):
        for state in range(len(states_kp1res)):
            var = Pkp1res_stat[res][state]
            dic[var] = Pkp1res_stat_val[res][state]
    return dic

def sub_inp( eqns, dic ):
    for eqn in range(len(eqns)):
        eqns[eqn] = eqns[eqn].subs(dic)
    return eqns

def lambdify_vec( eqns, var ):
    fncs = [lambdify((var), eqn, modules='numpy') for eqn in eqns]
    return fncs

def gen_fv( fncs ):
    return lambda zz: np.array([fnc(*zz) for fnc in fncs])

def gen_jv( fncs ):
    return lambda zz: np.array([ [ifnc(*zz) for ifnc in jfncs] for jfncs in fncs ])

def init_soln( N ):
    return 0.5*np.ones(N)

def sumsq_eqns( eqns, var ):
    sumsq = 0.
    for eqn in range(len(eqns)):
        sumsq += eqns[eqn].subs(var)**2
    return sumsq

# Set up the equations for the kres cond probs
def get_poss_states( Nres ):
    states = []
    state = [0]*Nres
    for gstate in range(2**Nres):
        states.append(state[:])
        state[0] = (state[0]+1) % 2
        for res in range(1,Nres):
            if ( (state[res-1]+1) % 2 == 1 ):
                state[res] = (state[res]+1) % 2
            else:
                break
    return states

def get_poss_states_endignore( Nres, Nend_front, Nend_back, end_state ):
    states = []
    state = [0]*Nres
    for gstate in range(2**Nres):
        state[0] = (state[0]+1) % 2
        for res in range(1,Nres):
            if ( (state[res-1]+1) % 2 == 1 ):
                state[res] = (state[res]+1) % 2
            else:
                break

        flag_add = True
        for res in range(Nend_front):
            if ( state[res] != end_state ):
                flag_add = False
        for res in range(Nend_back):
            if ( state[-(res+1)] != end_state ):
                flag_add = False

        if ( flag_add ):
            states.append(state[:])

    return states

# now, define the Nres variables
def define_PN(states_Nres, Nres, Ncorr):
    PN = []
    for s1 in range(len(states_Nres)):
        PN.append([])
        seq1 = ''.join(str(s) for s in states_Nres[s1])
        for s2 in range(len(states_Nres)):
            seq2 = ''.join(str(s) for s in states_Nres[s2])
            PN[s1].append(Symbol('P_'+seq2+'|'+seq1, nonnegative=True))
    return PN

# now, define the kp1res variables
def define_Pkp1res(states_kp1res, Nres, Ncorr):
    Pkp1res = []
    for res in range(Nres-Ncorr):
        Pkp1res.append([])
        for s1 in range(len(states_kp1res)):
            Pkp1res[res].append([])
            seq1 = ''.join(str(s) for s in states_kp1res[s1])
            for s2 in range(len(states_kp1res)):
                seq2 = ''.join(str(s) for s in states_kp1res[s2])
                res_str = ''
                for i in range(Ncorr+1):
                    res_str += str(res+i)+','
                res_str = res_str[:-1]
                Pkp1res[res][s1].append(Symbol('P^'+res_str+'_'+seq2+'|'+seq1, nonnegative=True))
    return Pkp1res

# also define the boundaries
def define_Pkp1res_end(states_kp1res, Nres, Ncorr):
    Pkp1res_end = []
    for res in range(2*Ncorr):
        Pkp1res_end.append([])
        for s1 in range(len(states_kp1res)):
            Pkp1res_end[res].append([])
            seq1 = ''.join(str(s) for s in states_kp1res[s1])
            for s2 in range(len(states_kp1res)):
                seq2 = ''.join(str(s) for s in states_kp1res[s2])
                poss = ''
                if ( res < Ncorr ):
                    for pos in range(0,Ncorr-res):
                        poss += 'b'+str(Ncorr-1-res-pos)+','
                    for pos in range(Ncorr-res,Ncorr+1):
                        poss += str(pos-(Ncorr-res))+','
                    poss = poss[:-1]
                else:
                    for pos in range(0,2*Ncorr-res):
                        poss += str(Nres-Ncorr+res-Ncorr+pos)
                    for pos in range(2*Ncorr-res,2*Ncorr):
                        poss += 'e'+str(pos-(2*Ncorr-res))+','
                    poss = poss[:-1]
                Pkp1res_end[res][s1].append( Symbol('P^'+poss+'_'+seq2+'|'+seq1, nonnegative=True) )
    return Pkp1res_end

# now, the conditional prob of jumping for each pair of res
def define_Pkres(states_kres, Nres, Ncorr):
    Pkres = []
    for res in range(Nres-(Ncorr-1)):
        Pkres.append([])
        for s1 in range(len(states_kres)):
            Pkres[res].append([])
            seq1 = ''.join(str(s) for s in states_kres[s1])
            for s2 in range(len(states_kres)):
                seq2 = ''.join(str(s) for s in states_kres[s2])
                res_str = ''
                for i in range(Ncorr):
                    res_str += str(res+i)+','
                res_str = res_str[:-1]
                Pkres[res][s1].append(Symbol('P^'+res_str+'_'+seq2+'|'+seq1, nonnegative=True))
    return Pkres

def define_Pkres_end(states_kres, Nres, Ncorr):
    # also define the boundaries
    Pkres_end = []
    for res in range(2*Ncorr):
        Pkres_end.append([])
        for s1 in range(len(states_kres)):
            Pkres_end[res].append([])
            seq1 = ''.join(str(s) for s in states_kres[s1])
            for s2 in range(len(states_kres)):
                seq2 = ''.join(str(s) for s in states_kres[s2])
                poss = ''
                if ( res < Ncorr ):
                    for pos in range(0,Ncorr-res):
                        poss += 'b'+str(Ncorr-1-res-pos)+','
                    for pos in range(Ncorr-res,Ncorr):
                        poss += str(pos-(Ncorr-res))+','
                    poss = poss[:-1]
                else:
                    for pos in range(1,2*Ncorr-res):
                        poss += str(Nres-Ncorr+res-Ncorr+pos)+','
                    for pos in range(2*Ncorr-res,2*Ncorr):
                        poss += 'e'+str(pos-(2*Ncorr-res))+','
                    poss = poss[:-1]
                Pkres_end[res][s1].append(Symbol('P^'+poss+'_'+seq2+'|'+seq1, nonnegative=True))
    return Pkres_end

# finally, the static probabilties of the kp1's and k's
def define_Pkp1res_stat(states_kp1res, Nres, Ncorr):
    Pkp1res_stat = []
    for res in range(Nres-Ncorr):
        Pkp1res_stat.append([])
        for s1 in range(len(states_kp1res)):
            seq1 = ''.join(str(s) for s in states_kp1res[s1])
            res_str = ''
            for i in range(Ncorr+1):
                res_str += str(res+i)+','
            res_str = res_str[:-1]
            Pkp1res_stat[res].append(Symbol('P^'+res_str+'_'+seq1, nonnegative=True))
    return Pkp1res_stat

# also define the boundaries
def define_Pkp1res_stat_end(states_kp1res, Nres, Ncorr):
    Pkp1res_stat_end = []
    for res in range(2*Ncorr):
        Pkp1res_stat_end.append([])
        for s1 in range(len(states_kp1res)):
            seq1 = ''.join(str(s) for s in states_kp1res[s1])
            poss = ''
            if ( res < Ncorr ):
                for pos in range(0,Ncorr-res):
                    poss += 'b'+str(Ncorr-1-res-pos)+','
                for pos in range(Ncorr-res,Ncorr+1):
                    poss += str(pos-(Ncorr-res))+','
                poss = poss[:-1]
            else:
                for pos in range(0,2*Ncorr-res):
                    poss += str(Nres-Ncorr+res-Ncorr+pos)+','
                for pos in range(2*Ncorr-res,2*Ncorr):
                    poss += 'e'+str(pos-(2*Ncorr-res))+','
                poss = poss[:-1]
            Pkp1res_stat_end[res].append(Symbol('P^'+poss+'_'+seq1, nonnegative=True))
    return Pkp1res_stat_end

def define_Pkres_stat(states_kres, Nres, Ncorr):
    Pkres_stat = []
    for res in range(Nres-(Ncorr-1)):
        Pkres_stat.append([])
        for s1 in range(len(states_kres)):
            seq1 = ''.join(str(s) for s in states_kres[s1])
            res_str = ''
            for i in range(Ncorr):
                res_str += str(res+i)+','
            res_str = res_str[:-1]
            Pkres_stat[res].append(Symbol('P^'+res_str+'_'+seq1, nonnegative=True))
    return Pkres_stat

# also define the boundaries
def define_Pkres_stat_end(states_kres, Nres, Ncorr):
    Pkres_stat_end = []
    for res in range(2*Ncorr):
        Pkres_stat_end.append([])
        for s1 in range(len(states_kres)):
            seq1 = ''.join(str(s) for s in states_kres[s1])
            poss = ''
            if ( res < Ncorr ):
                for pos in range(0,Ncorr-res):
                    poss += 'b'+str(Ncorr-1-res-pos)+','
                for pos in range(Ncorr-res,Ncorr):
                    poss += str(pos-(Ncorr-res))+','
                poss = poss[:-1]
            else:
                for pos in range(1,2*Ncorr-res):
                    poss += str(Nres-Ncorr+res-Ncorr+pos)+','
                for pos in range(2*Ncorr-res,2*Ncorr):
                    poss += 'e'+str(pos-(2*Ncorr-res))+','
                poss = poss[:-1]
            Pkres_stat_end[res].append(Symbol('P^'+poss+'_'+seq1, nonnegative=True))
    return Pkres_stat_end

# specify the boundary conditions
#eqns_bndry_cond = []
def get_bndry_cond_stat_dict(states_kp1res, states_kres, Nres, Ncorr, Pkp1res_stat_end, Pkres_stat, end_state):
    inp_bndry_cond_stat = {} # try using a dic for the bndry instead of solving the equations
    # now, relate the kp1res static prop ## JFR - This is quite confusing, I should double check it

    # the kp1 stat dist
    for res in range(2*Ncorr):
        for s1 in range(len(states_kp1res)):
            var = Pkp1res_stat_end[res][s1]
            tmp = 0.
            if ( res < Ncorr ):
                Nshift = res
                Nsum = (Ncorr-1)-Nshift
                Nfixed = Nshift+1
                if ( Nsum == 0 ):
                    ind = np.where( np.all(states_kres==states_kp1res[s1][1:],axis=1) == True )[0][0]
                    tmp += Pkres_stat[0][ind]
                else:
                    states = get_poss_states(Nsum)
                    for state in states:
                        state_full = np.hstack( (states_kp1res[s1][-Nfixed:],state) )
                        ind = np.where( np.all(states_kres==state_full,axis=1) == True )[0][0]
                        tmp += Pkres_stat[0][ind]
                end_fact = 1.
                for ends in range(Nsum+1):
                    if ( states_kp1res[s1][ends] == end_state ):
                        end_fact *= 1.
                    else:
                        end_fact *= 0.
                inp_bndry_cond_stat[var] = end_fact*tmp
            else:
                Nshift = 2*Ncorr-1-res
                Nsum = (Ncorr-1)-Nshift
                Nfixed = Nshift+1
                if ( Nsum == 0 ):
                    ind = np.where( np.all(states_kres==states_kp1res[s1][:-1],axis=1) == True )[0][0]
                    tmp += Pkres_stat[Nres-(Ncorr-1)-1][ind]
                else:
                    states = get_poss_states(Nsum)
                    for state in states:
                        state_full = np.hstack( (state,states_kp1res[s1][:Nfixed]) )
                        ind = np.where( np.all(states_kres==state_full,axis=1) == True )[0][0]
                        tmp += Pkres_stat[Nres-(Ncorr-1)-1][ind]
                end_fact = 1.
                for ends in range(Nsum+1):
                    if ( states_kp1res[s1][-1-ends] == end_state ):
                        end_fact *= 1.
                    else:
                        end_fact *= 0.
                inp_bndry_cond_stat[var] = end_fact*tmp
    return inp_bndry_cond_stat

def get_bndry_cond_dict(states_kp1res, states_kres, Nres, Ncorr, Pkp1res_end, Pkres, end_state):
    # and the kp1 res cond prob
    inp_bndry_cond = {} # separate the cond prob from the static prob boundary conditions
    for res in range(2*Ncorr):
        for s1 in range(len(states_kp1res)):
            for s2 in range(len(states_kp1res)):
                var = Pkp1res_end[res][s1][s2]
                tmp = 0.
                tmp_stat = 0.
                if ( res < Ncorr ):
                    Nshift = res
                    Nsum = (Ncorr-1)-Nshift
                    Nfixed = Nshift+1
                    if ( Nsum == 0 ):
                        ind = np.where( np.all(states_kres==states_kp1res[s1][1:],axis=1) == True )[0][0]
                        indp = np.where( np.all(states_kres==states_kp1res[s2][1:],axis=1) == True )[0][0]
                        tmp_stat = 1.
                        tmp += Pkres[0][ind][indp]
                    else:
                        states = get_poss_states(Nsum)
                        for state in states:
                            state_full = np.hstack( (states_kp1res[s1][-Nfixed:],state) )
                            ind = np.where( np.all(states_kres==state_full,axis=1) == True )[0][0]
                            tmp_stat += Pkres_stat[0][ind]
                            statesp = get_poss_states(Nsum)
                            for statep in statesp:
                                state_fullp = np.hstack( (states_kp1res[s2][-Nfixed:],statep) )
                                indp = np.where( np.all(states_kres==state_fullp,axis=1) == True )[0][0]
                                tmp += Pkres[0][ind][indp]*Pkres_stat[0][ind]
                    end_fact = 1.
                    for ends in range(Nsum+1):
                        if ( states_kp1res[s2][ends] == end_state ):
                            end_fact *= 1.
                        else:
                            end_fact *= 0.
                    inp_bndry_cond[var] = end_fact*tmp/tmp_stat
                else:
                    Nshift = 2*Ncorr-1-res
                    Nsum = (Ncorr-1)-Nshift
                    Nfixed = Nshift+1
                    if ( Nsum == 0 ):
                        ind = np.where( np.all(states_kres==states_kp1res[s1][:-1],axis=1) == True )[0][0]
                        indp = np.where( np.all(states_kres==states_kp1res[s2][:-1],axis=1) == True )[0][0]
                        tmp_stat = 1.
                        tmp += Pkres[Nres-(Ncorr-1)-1][ind][indp]
                    else:
                        states = get_poss_states(Nsum)
                        for state in states:
                            state_full = np.hstack( (state,states_kp1res[s1][:Nfixed]) )
                            ind = np.where( np.all(states_kres==state_full,axis=1) == True )[0][0]
                            tmp_stat += Pkres_stat[Nres-(Ncorr-1)-1][ind]
                            statesp = get_poss_states(Nsum)
                            for statep in statesp:
                                state_fullp = np.hstack( (statep,states_kp1res[s2][:Nfixed]) )
                                indp = np.where( np.all(states_kres==state_fullp,axis=1) == True )[0][0]
                                tmp += Pkres[Nres-(Ncorr-1)-1][ind][indp]*Pkres_stat[Nres-(Ncorr-1)-1][ind]
                    end_fact = 1.
                    for ends in range(Nsum+1):
                        if ( states_kp1res[s2][-1-ends] == end_state ):
                            end_fact *= 1.
                        else:
                            end_fact *= 0.
                    inp_bndry_cond[var] = end_fact*tmp/tmp_stat
    return inp_bndry_cond

def get_bndry_cond_kres_stat_dict(states_kp1res, states_kres, Nres, Ncorr, Pkres_stat_end, Pkres_stat, end_state):
    # and the kres static prop
    # Pkres stat, boundary only
    inp_bndry_cond_kres_stat = {}
    for res in range(2*Ncorr):
        for s1 in range(len(states_kres)):
            var = Pkres_stat_end[res][s1]
            tmp = 0.
            if ( (res==0) or (res==2*Ncorr-1) ):
                end_fact = 1.
                for ends in range(Ncorr):
                    if ( states_kp1res[s1][ends] == end_state ):
                        end_fact *= 1.
                    else:
                        end_fact *= 0.
                inp_bndry_cond_kres_stat[var] = end_fact
            elif ( res < Ncorr ):
                Nshift = res-1
                Nsum = (Ncorr-1)-Nshift
                Nfixed = Nshift+1
                if ( Nsum == 0 ):
                    raise ValueError('In boundary conditions for Pkres stat, Nsum==0, this should not happen!')
                else:
                    states = get_poss_states(Nsum)
                    for state in states:
                        state_full = np.hstack( (states_kres[s1][-Nfixed:],state) )
                        ind = np.where( np.all(states_kres==state_full,axis=1) == True )[0][0]
                        tmp += Pkres_stat[0][ind]
                end_fact = 1.
                for ends in range(Nsum):
                    if ( states_kres[s1][ends] == end_state ):
                        end_fact *= 1.
                    else:
                        end_fact *= 0.
                inp_bndry_cond_kres_stat[var] = end_fact*tmp
            else:
                Nshift = (2*Ncorr-1)-(res+1)
                Nsum = (Ncorr-1)-Nshift
                Nfixed = Nshift+1
                if ( Nsum == 0 ):
                    raise ValueError('In boundary conditions for Pkres stat, Nsum==0, this should not happen!')
                else:
                    states = get_poss_states(Nsum)
                    for state in states:
                        state_full = np.hstack( (state,states_kres[s1][:Nfixed]) )
                        ind = np.where( np.all(states_kres==state_full,axis=1) == True )[0][0]
                        tmp += Pkres_stat[Nres-(Ncorr-1)-1][ind]
                end_fact = 1.
                for ends in range(Nsum):
                    if ( states_kres[s1][-1-ends] == end_state ):
                        end_fact *= 1.
                    else:
                        end_fact *= 0.
                inp_bndry_cond_kres_stat[var] = end_fact*tmp
    return inp_bndry_cond_kres_stat

def get_bndry_cond_kres_dict(states_kp1res, states_kres, Nres, Ncorr, Pkres_end, Pkres, end_state):
    # and the kres cond prob
    inp_bndry_cond_kres = {} # separate the cond prob from the static prob boundary conditions
    for res in range(2*Ncorr):
        for s1 in range(len(states_kres)):
            for s2 in range(len(states_kres)):
                var = Pkres_end[res][s1][s2]
                tmp = 0.
                tmp_stat = 0.
                if ( (res==0) or (res==2*Ncorr-1) ):
                    end_fact = 1.
                    for ends in range(Ncorr):
                        if ( states_kres[s2][ends] == end_state ):
                            end_fact *= 1.
                        else:
                            end_fact *= 0.
                    inp_bndry_cond_kres[var] = end_fact
                elif ( res < Ncorr ):
                    Nshift = res-1
                    Nsum = (Ncorr-1)-Nshift
                    Nfixed = Nshift+1
                    if ( Nsum == 0 ):
                        raise ValueError('In boundary conditions for Pkres, Nsum==0, this should not happen!')
                    else:
                        states = get_poss_states(Nsum)
                        for state in states:
                            state_full = np.hstack( (states_kres[s1][-Nfixed:],state) )
                            ind = np.where( np.all(states_kres==state_full,axis=1) == True )[0][0]
                            tmp_stat += Pkres_stat[0][ind]
                            statesp = get_poss_states(Nsum)
                            for statep in statesp:
                                state_fullp = np.hstack( (states_kres[s2][-Nfixed:],statep) )
                                indp = np.where( np.all(states_kres==state_fullp,axis=1) == True )[0][0]
                                tmp += Pkres[0][ind][indp]*Pkres_stat[0][ind]
                    end_fact = 1.
                    for ends in range(Nsum):
                        if ( states_kres[s2][ends] == end_state ):
                            end_fact *= 1.
                        else:
                            end_fact *= 0.
                    inp_bndry_cond_kres[var] = end_fact*tmp/tmp_stat
                else:
                    Nshift = (2*Ncorr-1)-(res+1)
                    Nsum = (Ncorr-1)-Nshift
                    Nfixed = Nshift+1
                    if ( Nsum == 0 ):
                        raise ValueError('In boundary conditions for Pkres, Nsum==0, this should not happen!')
                    else:
                        states = get_poss_states(Nsum)
                        for state in states:
                            state_full = np.hstack( (state,states_kres[s1][:Nfixed]) )
                            ind = np.where( np.all(states_kres==state_full,axis=1) == True )[0][0]
                            tmp_stat += Pkres_stat[Nres-(Ncorr-1)-1][ind]
                            statesp = get_poss_states(Nsum)
                            for statep in statesp:
                                state_fullp = np.hstack( (statep,states_kres[s2][:Nfixed]) )
                                indp = np.where( np.all(states_kres==state_fullp,axis=1) == True )[0][0]
                                tmp += Pkres[Nres-(Ncorr-1)-1][ind][indp]*Pkres_stat[Nres-(Ncorr-1)-1][ind]
                    end_fact = 1.
                    for ends in range(Nsum):
                        if ( states_kres[s2][-1-ends] == end_state ):
                            end_fact *= 1.
                        else:
                            end_fact *= 0.
                    inp_bndry_cond_kres[var] = end_fact*tmp/tmp_stat
    return inp_bndry_cond_kres


def get_eqns_Pkres(states_kp1res, states_kres, Nres, Ncorr, Pkp1res, Pkp1res_end, Pkp1res_stat, Pkp1res_stat_end, Pkres, Pkres_end, Pkres_stat, Pkres_stat_end, end_state):
    # moved the eqns_Pkres setup to after the bndry conditions
    eqns_Pkres = []
    Nk = 2*Ncorr
    Nres_corr = Ncorr+Nk
    state_ctr = 0
    for res in range(Nres-(Ncorr-1)):
        Nxxp = [[0.]*len(states_kres) for _ in range(len(states_kres))]
        #states = get_poss_states(Nres_corr)
        # only consider states with a fixed boundary condition
        if ( res < Ncorr+1 ):
            Nend_front = Ncorr - res
        else:
            Nend_front = 0
        resmax = (Nres-1)-(Ncorr-1)
        if ( res > resmax - Ncorr - 1 ):
            Nend_back = Ncorr - (resmax - res)
        else:
            Nend_back = 0
        states = get_poss_states_endignore( Nres_corr, Nend_front, Nend_back, end_state )
        state_ctr = 0
        for state in states:
            print 'starting state '+str(state_ctr)+' of '+str(len(states))+' for res '+str(res)
            state_ctr += 1
            #statesp = get_poss_states(Nres_corr)
            statesp = get_poss_states_endignore( Nres_corr, Nend_front, Nend_back, end_state )
            for statep in statesp:
                tmp = 1.
                for kset in range(Nk):
                    indp = np.where( np.all(states_kp1res==statep[kset:kset+Ncorr+1],axis=1) == True )[0][0]
                    ind = np.where( np.all(states_kp1res==state[kset:kset+Ncorr+1],axis=1) == True )[0][0]
                    ind2p = np.where( np.all(states_kres==statep[kset:kset+Ncorr],axis=1) == True )[0][0]
                    ind2 = np.where( np.all(states_kres==state[kset:kset+Ncorr],axis=1) == True )[0][0]
                    if ( (res<Ncorr) and (kset<Ncorr-res) ):
                        tmp *= Pkp1res_end[res+kset][ind][indp]*Pkp1res_stat_end[res+kset][ind]
                        if ( (kset > 0) and (kset != Ncorr) ):
                            tmp /= (Pkres_end[res+kset][ind2][ind2p]*Pkres_stat_end[res+kset][ind2])
                    elif ( (res>Nres-Nk) and (kset>(Nk-1)-( res-(Nres-Nk+1)+1 )) ): # nb - the second term is kmax-Nksets  
                        # some useful quantities - old!
                        # pos_ind = pos_start + kset - kmin
                        # Nksets = res-(Nres-Ncorr-1)+1, 
                        # pos_start = Nres-1-Ncorr
                        # dkset = kset - ksetmax, posmax = Nres-Ncorr, dposmax = posmax-res
                        # ksetmax = Nk-1, kmin = kmax-Nksets+1
                        # some useful quantities - generalized for > 2 corr
                        # pos_start = Nres-Nk+1
                        # Nksets = res - pos_start + 1
                        # pos_ind = pos_start + kset - kmin
                        # dkset = kset - ksetmax, posmax = Nres-Ncorr, dposmax = posmax-res
                        # ksetmax = Nk-1, kmin = kmax-Nksets+1
                        # some useful quantities - generalized for > 2 corr
                        pos_start = (Nres-Nk+1)
                        Nksets = (res-pos_start+1)
                        kmax = (Nk-1)
                        endind_start = (kmax+1-Ncorr)
                        kmin = ( kmax - Nksets + 1 )
                        kmid = Ncorr
                        pos_ind = endind_start + kset - kmin
                        tmp *= Pkp1res_end[pos_ind][ind][indp]*Pkp1res_stat_end[pos_ind][ind]
                        if ( kset != kmid ):
                            if ( res+kset > Nres ):
                                tmp /= (Pkres_end[pos_ind-1][ind2][ind2p]*Pkres_stat_end[pos_ind-1][ind2]) # I am pretty sure about the pos_ind-1, but should check!
                            else:
                                tmp /= (Pkres[res-Ncorr+kset][ind2][ind2p]*Pkres_stat[res-Ncorr+kset][ind2])
                    else:
                        tmp *= Pkp1res[res-Ncorr+kset][ind][indp]*Pkp1res_stat[res-Ncorr+kset][ind]
                        if ( (kset!=0) and (kset!=Ncorr) ):
                            tmp /= (Pkres[res-Ncorr+kset][ind2][ind2p]*Pkres_stat[res-Ncorr+kset][ind2])
    
                ind = np.where( np.all(states_kres==state[Ncorr:Ncorr+Ncorr],axis=1) == True )[0][0]
                indp = np.where( np.all(states_kres==statep[Ncorr:Ncorr+Ncorr],axis=1) == True )[0][0]
                # let's try making all the substitutions on the fly
                #tmp = tmp.subs(inp_bndry_cond)
                #tmp = tmp.subs(inp_bndry_cond_stat)
                #tmp = tmp.subs(inp_bndry_cond_kres)
                #tmp = tmp.subs(inp_bndry_cond_kres_stat)
                #tmp = tmp.subs(inp_var)
                Nxxp[ind][indp] += tmp
        for kstate in range(len(states_kres)):
            den = Pkres_stat[res][kstate]
            for kstatep in range(len(states_kres)):
                eqns_Pkres.append( Pkres[res][kstate][kstatep] - (Nxxp[kstate][kstatep]**(0.5))/den )
    return eqns_Pkres
