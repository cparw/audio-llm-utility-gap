import sys, numpy as np, time
sys.path.insert(0, '.')
from vlib import *
t=time.time()
L = read_csv('scores/part22/edaic_lifted_perclip.csv')
F = {r['pid']: r for r in read_csv('scores/part16/EDAICFULL/edaic_full_perclip.csv')}
pid=[r['pid'] for r in L]; y=np.array([int(r['label']) for r in L]); s=np.array([float(r['p_yes_lifted']) for r in L])
d=np.array([float(F[p]['p_yes_answer']) for p in pid]); assert all(int(F[p]['label'])==yy for p,yy in zip(pid,y))
print('lifted', boot_single(y,s,pid), auc_trap(y,s))
print('default', boot_single(y,d,pid), auc_trap(y,d))
print('diff', boot_paired(y,s,d,pid))
T = read_csv('scores/part17/T7b/T7b_edaic300_meanof5_perclip.csv')
pid=[r['pid'] for r in T]; y=np.array([int(r['label']) for r in T]); a=np.array([float(r['p_yes_answer']) for r in T])
for k in ['enc','llm','ans','proj']:
    S5=[np.array([float(r[f'oof_{k}_r{j}']) for r in T]) for j in range(5)]
    print(k, [round(auc_mw(y,v),4) for v in S5], boot_mean5(y,S5,pid), boot_paired(y,S5,a,pid))
print(time.time()-t)
