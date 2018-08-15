import numpy as np
import cPickle as cp
from hera_cal import redcal

print 'Initializing variables..'
fqs  = .15 #GHz
lsts = np.pi/4
Nsim = 2**16
Nant = 37 

# Setup all variables
ants   = np.loadtxt('antenna_positions_37.dat')
antpos = {k:v for k,v in zip(range(37),ants)}
reds   = redcal.get_reds(antpos)
subreds = [reds[0], reds[1], reds[2]]

input_gains = {}
allbl_gains = {}
subbl_gains = {}

for a in range(Nant):
    input_gains[(a,'x')] = []
    allbl_gains[(a,'x')] = []
    subbl_gains[(a,'x')] = []

# Run simulation

print 'Starting simulation..'
for i in range(Nsim):
    print i
    gains, vis, data = redcal.sim_red_data(reds, shape=(1,1))
    data = {k:v+0.5*redcal.noise((1,1)) for k,v in data.items()}

    redcalibrator = redcal.RedundantCalibrator(reds)
    sol_degen = redcalibrator.logcal(data)
    sol = redcalibrator.remove_degen(antpos, sol_degen, degen_sol=gains)

    subredcalib = redcal.RedundantCalibrator(subreds)
    subsol_degen = subredcalib.logcal(data)
    subsol = subredcalib.remove_degen(antpos, subsol_degen, degen_sol=gains)

    for a in range(Nant):
        input_gains[(a,'x')].append(gains[(a,'x')][0][0])
        subbl_gains[(a,'x')].append(subsol[(a,'x')][0][0])    
        allbl_gains[(a,'x')].append(sol[(a,'x')][0][0])


# Write simulation to disk
with open('sim_gain_sols.cp','w') as fp:
     cp.dump([input_gains, subbl_gains, allbl_gains],fp,protocol=2)
