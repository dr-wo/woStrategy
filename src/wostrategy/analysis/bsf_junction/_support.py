"""Frozen 4G covariate matcher used by projected_mean. No outcome matching."""
from __future__ import annotations
import functools, hashlib, itertools, json
import numpy as np
import pandas as pd
LADDER=[(0,0),(1,0),(2,0),(4,0),(4,1),(4,2)]
LABELS=['L0_exact','L1_age1','L2_age2','L3_age4','L4_age4_lap1','L5_age4_lap2']
def encode(value):return json.dumps(value,separators=(',',':'))
def match_pair(a,b,age_radius,lap_radius,clock_limit=None):
    """Maximum-cardinality order-preserving 1:1 match; min total lap distance,
    then total physical-age distance, then lexicographic source row IDs.
    Pair ordering forbids time reversal. Clock separation is diagnostic except
    for the explicitly named clock-limit sensitivity.
    """
    aa=list(a.sort_values(['LapNumber','row_id']).itertuples()); bb=list(b.sort_values(['LapNumber','row_id']).itertuples())
    @functools.lru_cache(None)
    def solve(i,j):
        if i==len(aa) or j==len(bb): return (0,0,0,())
        opts=[solve(i+1,j),solve(i,j+1)]
        x,y=aa[i],bb[j]
        dl=abs(int(x.LapNumber-y.LapNumber)); da=abs(int(x.TyreLife-y.TyreLife))
        if dl<=lap_radius and da<=age_radius and (clock_limit is None or abs(x.clock_seconds-y.clock_seconds)<=clock_limit):
            n,sl,sa,p=solve(i+1,j+1)
            opts.append((n+1,sl+dl,sa+da,((int(x.row_id),int(y.row_id)),)+p))
        return min(opts,key=lambda v:(-v[0],v[1],v[2],v[3]))
    return list(solve(0,0)[3])

def build_edges(frames,threshold=4,window_offset=0,clock_limit=None):
    records=[]
    for r,d0 in frames.items():
        d=d0.copy(); d['window']=((d.LapNumber-1-window_offset)//10).astype(int)
        index=d.set_index('row_id')
        for (compound,fresh,window),group in d.groupby(['Compound','FreshTyre','window'],sort=True):
            nodes=[(key,g) for key,g in group.groupby(['Team','Driver','Stint'],sort=True) if len(g)>=threshold]
            for (ka,ga),(kb,gb) in itertools.combinations(nodes,2):
                if ka[0]==kb[0]: continue
                for level,(age_radius,lap_radius) in enumerate(LADDER):
                    pairs=match_pair(ga,gb,age_radius,lap_radius,clock_limit)
                    if len(pairs)<threshold: continue
                    cells=[]
                    for ia,ib in pairs:
                        x,y=index.loc[ia],index.loc[ib]
                        cells.append(dict(a_row=int(ia),b_row=int(ib),a_lap=int(x.LapNumber),b_lap=int(y.LapNumber),a_life=int(x.TyreLife),b_life=int(y.TyreLife),a_age=int(x.TyreAgeLaps),b_age=int(y.TyreAgeLaps),a_clock=float(x.clock_seconds),b_clock=float(y.clock_seconds),lap_distance=int(abs(x.LapNumber-y.LapNumber)),life_distance=int(abs(x.TyreLife-y.TyreLife)),clock_distance=float(abs(x.clock_seconds-y.clock_seconds))))
                    identity=[threshold,r,compound,bool(fresh),int(window),list(ka),list(kb),window_offset,clock_limit]
                    rec=dict(edge_id='G'+hashlib.sha256(encode(identity).encode()).hexdigest()[:16],round=r,event=d.EventName.iloc[0],threshold=threshold,level=level,level_label=LABELS[level],compound=compound,fresh=bool(fresh),window=int(window),window_start=int(window)*10+1+window_offset,window_end=int(window)*10+10+window_offset,a_team=ka[0],a_driver=ka[1],a_stint=float(ka[2]),b_team=kb[0],b_driver=kb[1],b_stint=float(kb[2]),n=len(cells),pairs=pairs,cells=cells,age_radius=age_radius,lap_radius=lap_radius)
                    for name,col in [('life','life_distance'),('lap','lap_distance'),('clock','clock_distance')]:
                        val=[c[col] for c in cells]
                        rec[name+'_distance_sum']=float(sum(val)); rec[name+'_distance_mean']=float(np.mean(val));rec[name+'_distance_max']=float(max(val))
                    rec['clock_within60_n']=sum(c['clock_distance']<=60 for c in cells)
                    records.append(rec)
                    break
    return records
