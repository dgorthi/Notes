import numpy as np
import matplotlib.pyplot as plt
import hera_cal
import omnical
import h5py
import aipy
import cPickle as cp
from hera_cal.utils import AntennaArray
from collections import OrderedDict

def create_aa(antpos,fqs):
    nants = len(antpos.keys())
    location = (-0.5362, 0.37399, 1051.6900)

    # get antenna positions from file

    # make antpos_ideal array
    antpos_ideal = np.zeros(shape=(nants, 3), dtype=float) - 1

    # unpack from dict -> numpy array
    for k in antpos.keys():
        antpos_ideal[k, :] = np.array([antpos[k][0], antpos[k][1], antpos[k][2]])

    # Make list of antennas.
    # These are values for a zenith-pointing antenna, with a dummy Gaussian beam.
    antennas = []
    for i in range(nants):
        beam = aipy.fit.Beam(fqs)
        phsoff = {'x': [0., 0.], 'y': [0., 0.]}
        amp = 1.
        amp = {'x': amp, 'y': amp}
        bp_r = [1.]
        bp_r = {'x': bp_r, 'y': bp_r}
        bp_i = [0., 0., 0.]
        bp_i = {'x': bp_i, 'y': bp_i}
        twist = 0.
        antennas.append(aipy.pol.Antenna(0., 0., 0., beam, phsoff=phsoff,
                                         amp=amp, bp_r=bp_r, bp_i=bp_i, pointing=(0., np.pi / 2, twist)))

    # Make the AntennaArray and set position parameters
    aa = AntennaArray(location, antennas, antpos_ideal=antpos_ideal)

    pos_prms = {}
    for k,v in antpos.items():
        pos_prms[str(k)] = {'x':v[0], 'y':v[1], 'z':v[2]}
    aa.set_params(pos_prms)
    return aa

Nants = 37
locs = np.loadtxt('antenna_positions_%d.dat'%Nants)
antpos = {}
for pos,ant in zip(locs, np.arange(Nants)):
    antpos[ant] = pos

redbls = hera_cal.redcal.get_pos_reds(antpos)

fqs = np.linspace(.1,.2,num=1024,endpoint=False)
lsts = np.linspace(0,2*np.pi,num=100,endpoint=False)
times = lsts/(2*np.pi)*aipy.const.sidereal_day

gains = OrderedDict(); gains['x'] = OrderedDict()
vis = OrderedDict(); 
wgts = OrderedDict(); hdr = {}

with h5py.File('fake_vis.hdf5','r') as fp:
    for key in fp.keys():
        if (key.rfind('-') > 0):
            a1, a2 = key.split('-')
            vis[(int(a1),int(a2))] = OrderedDict()
            vis[(int(a1),int(a2))]['xx'] = fp[key][...]
            wgts[(int(a1),int(a2))] = OrderedDict()
            wgts[(int(a1),int(a2))]['xx'] = np.ones([len(lsts),len(fqs)])
        elif (key.startswith('g')):
            k = key.lstrip('g')
            gains['x'][int(k)] = np.repeat([fp[key][...]],100,axis=0)
        else: hdr[key] = fp[key][...]

# ------ All baselines ---------
aa = create_aa(antpos,fqs)
#info = hera_cal.omni.aa_to_info(aa, pols=['x'], fcal=True)
#fc = hera_cal.firstcal.FirstCal(vis,wgts,fqs,info)
#delays = fc.run()
#
#with open('delay_sols.cp','w') as fp:
#    cp.dump(delays, fp, protocol=2)

# Flat gain priors to run omnical
g0 = OrderedDict(); g0['x'] = OrderedDict()
for pair in gains['x'].keys():
    g0['x'][pair] = np.ones([len(lsts), len(fqs)])

#print 'Running omnical'
#info = hera_cal.omni.aa_to_info(aa, pols=['x'])
#m1, g1, v1 = omnical.calib.logcal(vis, info, xtalk=None, gains=g0,
#                                  maxiter=50, conv=1e-3, stepsize=.3,
#                                  trust_period=1)
#
#m2, g2, v2 = omnical.calib.lincal(vis, info, gains=g1, vis=v1, xtalk=None,
#                                  conv=1e-3, stepsize=.3,
#                                  trust_period=1, maxiter=50)
#
#g3, v3 = hera_cal.omni.remove_degen(info, g2, v2, gains, minV=False)
#
#with open('fake_vis_sols.cp','w') as fp:
#    cp.dump([m2,g3,v3],fp,protocol=2)

subbls = [redbls[0], redbls[1], redbls[2]]
subbls_list = [bl for reds in subbls for bl in reds]
subvis = OrderedDict();
for reds in subbls:
    for bl in reds:
        subvis[bl] = vis[bl]
        
info = hera_cal.omni.aa_to_info(aa, pols=['x'],bls=subbls_list)
print ('Running Logcal')
m1, g1, v1 = omnical.calib.logcal(subvis, info, xtalk=None, gains=g0,
                                  maxiter=500, conv=1e-3, stepsize=.03,
                                  trust_period=1)
print ('Running lincal')
m2, g2, v2 = omnical.calib.lincal(subvis, info, gains=g1, vis=v1, xtalk=None,
                                  conv=1e-3, stepsize=.3,
                                  trust_period=1, maxiter=50)
print ('Removing degeneracies')
g3, v3 = hera_cal.omni.remove_degen(info, g2, v2, gains, minV=False)

with open('fake_vis_sub_sols_smallerstep.cp','w') as fp:
    cp.dump([m2,g3,v3],fp,protocol=2)
