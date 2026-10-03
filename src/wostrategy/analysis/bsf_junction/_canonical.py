"""Read-only frozen canonical reference functional over persisted producer samples.

This computes the R&D projection's posterior mean; it never changes the
producer's candidates, parameters, weights, baseline summaries or tracker.
"""
import json
import numpy as np
import pandas as pd
from .constants import TEAMS
class CanonicalBank:
 def __init__(self,d,source_paths):
  self.d=d.reset_index(drop=True).copy()
  self.teams=TEAMS;self.tidx={t:j for j,t in enumerate(TEAMS)};self.codes=np.array([self.tidx[t] for t in self.d.Team])
  self.source_paths=source_paths
  params=pd.read_csv(source_paths['parameters']).sort_values('SampleId').reset_index(drop=True)
  N=len(params)
  self.n_samples=N
  if not N or params.SampleId.nunique()!=N: raise ValueError('Expected unique saved candidates')
  if not (params.WeightStrategy.eq('gaussian').all() and params.BaselineGroup.eq('team').all()):
   raise ValueError('Frozen BSF reference requires Gaussian team-baseline producer candidates')
  self.params=params;self.fuel=params.FuelRateSecondsPerLap.to_numpy(float);self.track=params.TrackRateSecondsPerLap.to_numpy(float)
  self.fuel_ref=params.FuelRef.to_numpy(float);self.race_ref=params.RaceLapRef.to_numpy(float);self.age_ref=params.TyreAgeRef.to_numpy(float)
  if not all(np.isfinite(z).all() and np.ptp(z)==0 for z in [self.fuel_ref,self.race_ref,self.age_ref]):
   raise ValueError('Saved canonical references must be finite and constant across candidates')
  delta=pd.read_csv(source_paths['delta'],usecols=['SampleId','Compound','CompoundDeltaSeconds'])
  deg=pd.read_csv(source_paths['team_deg'],usecols=['SampleId','Team','Compound','TeamCompoundDegSecondsPerLap'])
  self.delta={};self.deg={}
  for comp,g in delta.groupby('Compound'):
   g=g.sort_values('SampleId')
   if not np.array_equal(g.SampleId.to_numpy(),params.SampleId.to_numpy()):raise ValueError('Compound candidate IDs differ from parameter bank')
   self.delta[str(comp)]=g.CompoundDeltaSeconds.to_numpy(float)
  for pair,g in deg.groupby(['Team','Compound']):
   g=g.sort_values('SampleId')
   if not np.array_equal(g.SampleId.to_numpy(),params.SampleId.to_numpy()):raise ValueError('Degradation candidate IDs differ from parameter bank')
   self.deg[tuple(map(str,pair))]=g.TeamCompoundDegSecondsPerLap.to_numpy(float)
  self.y=self.d.LapTimeSeconds.to_numpy(float);self.count=np.bincount(self.codes,minlength=11)
  self.anchors=np.array([self.y[self.codes==j].mean() if self.count[j] else 0 for j in range(11)])
  self.sums=np.zeros((N,11));self.squares=np.zeros((N,11))
  for start in range(0,N,5000):
   sl=slice(start,start+5000);res=self.y[None,:]-self.anchors[self.codes][None,:]-self.effect(self.d,sl)
   for j in range(11):
    x=res[:,self.codes==j];self.sums[sl,j]=x.sum(axis=1);self.squares[sl,j]=np.square(x).sum(axis=1)
  self.sses=self.squares-np.divide(self.sums**2,self.count[None,:],out=np.zeros_like(self.sums),where=self.count[None,:]>0)
  self.full_sse=self.sses.sum(axis=1)
  meta=json.loads(source_paths['metadata'].read_text());self.sigma=float(meta['clean_lap_noise_sigma'])
  if self.sigma!=.5:raise ValueError('Frozen BSF reference requires clean-lap noise sigma 0.5')

 def effect(self,d,sl=slice(None)):
  out=self.fuel[sl,None]*(d.FuelProxyLapsRemaining.to_numpy()[None,:]-self.fuel_ref[sl,None])+self.track[sl,None]*(d.LapNumber.to_numpy()[None,:]-self.race_ref[sl,None])
  ts=d.Team.to_numpy();cs=d.Compound.to_numpy();age=d.TyreAgeLaps.to_numpy()
  for pair in set(zip(ts,cs)):
   mask=(ts==pair[0])&(cs==pair[1]);out[:,mask]+=self.deg[pair][sl,None]*(age[mask][None,:]-self.age_ref[sl,None])+self.delta[pair[1]][sl,None]
  return out

 def fit(self,excluded=()):
  held=np.unique(np.asarray(excluded,dtype=int).ravel())
  if len(held) and (held.min()<0 or held.max()>=len(self.d)):raise IndexError('Raw row ID out of range')
  counts=self.count-np.bincount(self.codes[held],minlength=11)
  if counts.sum()==0:raise ValueError('No training rows remain')
  sums=self.sums.copy();squares=self.squares.copy()
  if len(held):
   hd=self.d.iloc[held]
   for start in range(0,self.n_samples,5000):
    sl=slice(start,start+5000);res=self.y[held][None,:]-self.anchors[self.codes[held]][None,:]-self.effect(hd,sl)
    for j in np.unique(self.codes[held]):
     x=res[:,self.codes[held]==j];sums[sl,j]-=x.sum(axis=1);squares[sl,j]-=np.square(x).sum(axis=1)
  base=np.divide(sums,counts[None,:],out=np.full_like(sums,np.nan),where=counts[None,:]>0)+self.anchors[None,:]
  sse=(squares-np.divide(sums*sums,counts[None,:],out=np.zeros_like(sums),where=counts[None,:]>0))[:,counts>0].sum(axis=1)
  logw=-np.maximum(sse,0)/counts.sum()/(2*self.sigma**2);w=np.exp(logw-logw.max());w/=w.sum()
  return CanonicalFit(self,held,counts,base,w,sse)

class CanonicalFit:
 def __init__(self,bank,excluded,counts,base,weights,sse):
  self.bank=bank;self.excluded=excluded;self.counts=counts;self.base=base;self.weights=weights;self.sse=sse
  self.ess=float(1/np.square(weights).sum());self.point_base=weights@base
  self.fuel=float(weights@bank.fuel);self.track=float(weights@bank.track)
  self.delta={k:float(weights@v) for k,v in bank.delta.items()};self.deg={k:float(weights@v) for k,v in bank.deg.items()}

 def predict(self,d):
  b=self.bank
  return np.array([self.point_base[b.tidx[x.Team]]+self.fuel*(x.FuelProxyLapsRemaining-b.fuel_ref[0])+self.track*(x.LapNumber-b.race_ref[0])+self.deg[(x.Team,x.Compound)]*(x.TyreAgeLaps-b.age_ref[0])+self.delta[x.Compound] for x in d.itertuples()])

 def predict_rows(self,ids):
  ids=np.asarray(ids,dtype=int);return self.predict(self.bank.d.iloc[ids.ravel()]).reshape(ids.shape)

 def predict_pairs(self,ids):
  ids=np.asarray(ids,dtype=int);assert ids.ndim==2 and ids.shape[1]==2
  p=self.predict_rows(ids);return p[:,0]-p[:,1]
