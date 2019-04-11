import hera_cal.redcal as om
import hera_cal.omni
import omnical.calib
import numpy as np
import time

np.random.seed(0)
SHAPE = (10,2048)
NOISE = 1e-3

for Nants in [37, 128, 243, 350]:

    ants = np.loadtxt('antenna_positions_%d.dat'%Nants)
    idxs = np.arange(Nants)
    antpos = {}
    for k,v in zip(idxs,ants):
        antpos[k] = v

    reds = om.get_reds(antpos, pols=['xx'], pol_mode='1pol')

    info = om.RedundantCalibrator(reds)

    gains, true_vis, d = om.sim_red_data(reds, shape=SHAPE, gain_scatter=.01)
    d = {key: value.astype(np.complex64) for key,value in d.items()}
    d_nos = {key: value + NOISE * om.noise(value.shape) for key,value in d.items()}
    d_nos = {key: value.astype(np.complex64) for key,value in d_nos.items()}

    w = dict([(k, np.float32(1.)) for k in d.keys()])
    sol0 = dict([(k, np.ones_like(v)) for k, v in gains.items()])
    sol0.update(info.compute_ubls(d, sol0))
    sol0 = {k:v.astype(np.complex64) for k,v in sol0.items()}
    
    print('NANTS: %d'%Nants)
    
    t0 = time.time()
    meta_omnical, sol_omnical = info.omnical(d, sol0, 
                                             gain=.5, maxiter=500, check_after=30, check_every=6)
    print('OMNICAL no noise: %4.1f s' % (time.time() - t0), 
          sol_omnical.values()[0].dtype)

    t0 = time.time()
    meta_nos_omnical, sol_nos_omnical = info.omnical(d_nos, sol0, 
                                                     gain=.5, maxiter=500, check_after=30, check_every=6)
    print('OMNICAL with noise: %4.1f s\n\n' % (time.time() - t0), 
          sol_nos_omnical.values()[0].dtype)
