import json,csv,os,statistics as st,collections
import cha
B="/Volumes/G-Drive Pro/paper work/Hard drive data for paper"; DB=B+"/DementiaBank"
A20=DB+"/challenges/ADReSS-2020"
D=json.load(open('part1a.json'))
def inter(u,w0,w1):
    return max(0.0, min(u['end_ms']/1000.0,w1)-max(u['start_ms']/1000.0,w0))
out={}
for key,recs,chafn in [('pitt',D['pitt'],lambda r:r['cha']),('a20',D['a20'],lambda r:r['cha'])]:
    for r in recs:
        u=cha.parse_cha(chafn(r))
        inv=sum(inter(x,r['w0'],r['w1']) for x in u if x['who']=='INV')
        par=sum(inter(x,r['w0'],r['w1']) for x in u if x['who']=='PAR')
        r['inv_s_in']=inv; r['par_s_in']=par
    print(key,'inv_s_in>0:',sum(1 for r in recs if r['inv_s_in']>0),
          '>0.5s:',sum(1 for r in recs if r['inv_s_in']>0.5),
          '>2s:',sum(1 for r in recs if r['inv_s_in']>2.0),
          'median inv_s',round(st.median([r['inv_s_in'] for r in recs]),4),
          'max inv_s',round(max(r['inv_s_in'] for r in recs),4))
    print('   inv_ov words>0:',sum(1 for r in recs if r['inv_ov']>0),
          'inv fully-inside words>0:',sum(1 for r in recs if r['inv_in']>0))
    print('   PAR overlap words <15:',sum(1 for r in recs if r['par_ov']<15),
          'PAR fully-inside <15:',sum(1 for r in recs if r['par_in']<15))
json.dump(D,open('part1a.json','w'))
