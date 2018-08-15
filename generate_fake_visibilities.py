import numpy as np
import matplotlib.pyplot as plt
import aipy, uvtools
from hera_sim import foregrounds, noise, sigchain, rfi
from hera_cal import redcal 
import cPickle as cp
import h5py

fqs = np.linspace(.1,.2,num=1024,endpoint=False)
lsts = np.linspace(0,np.pi/4,10,endpoint=False) 
#times = lsts/(2*np.pi) * aipy.const.sidereal_day

def get_distance(bl,antpos):
    """
    Given a tuple of antennas and positions of the antennas 
    returns the distance between the two antennas.
    """
    x1,y1,z1 = antpos[bl[0]]
    x2,y2,z2 = antpos[bl[1]]
    dist = np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
    return dist

## Generate visibilities for all redundant baselines in array.
ants = np.loadtxt('antenna_positions_37.dat')
idxs = np.arange(37)
antpos = {}

for k,v in zip(idxs, ants):
    antpos[k] = v

reds = redcal.get_pos_reds(antpos)

# Extract all ants
ants = list(set([ant for bls in reds for bl in bls for ant in bl]))

# Generate gains
gains = sigchain.gen_gains(fqs, ants, dly_rng=(-1,1))
true_vis, data = {}, {}

# Generate sky model--common for all ants
Tsky_mdl = noise.HERA_Tsky_mdl['xx']
tsky = noise.resample_Tsky(fqs,lsts,Tsky_mdl=noise.HERA_Tsky_mdl['xx'])

fp = h5py.File('fake_vis.hdf5','a')

fp.attrs.create('Nants', 37)
fp.create_dataset('fqs',data=fqs)
fp.create_dataset('lsts',data=lsts)

for a in ants:
    fp.create_dataset('g%d'%a,data=gains[a])

for bls in reds:
    for (i,j) in bls:
        data[(i,j)] = fp.create_dataset('%d-%d'%(i,j),(len(lsts),len(fqs)),chunks=True,dtype='complex64')

for bls in reds:
    bl_len = get_distance(bls[0],antpos)
    bl_len_ns = bl_len/aipy.const.c * 1e11
    vis_fg_pntsrc = foregrounds.pntsrc_foreground(lsts, fqs, bl_len_ns, nsrcs=200)
    vis_fg_diffuse = foregrounds.diffuse_foreground(Tsky_mdl,lsts,fqs,bl_len_ns)
    true_vis[bls[0]] = vis_fg_pntsrc + vis_fg_diffuse

    for (i,j) in bls:
        print (i,j)
        nos_jy = noise.sky_noise_jy(tsky+150., fqs, lsts)
        vis_tot = nos_jy + true_vis[bls[0]]
        data[(i,j)][:,:] = sigchain.apply_gains(vis_tot, gains, (i,j))

fp.close()
