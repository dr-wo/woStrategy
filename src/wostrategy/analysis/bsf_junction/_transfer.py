"""Exact complete-schedule donor means, copied from the frozen R&D-4H kernel."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
COLS=['Team','Driver','Stint','Compound','FreshTyre','LapNumber','TyreLife','TyreAgeLaps','LapStartTime']
def describe_support(d):
    """Only listed covariates are read; LapTimeSeconds can be absent or NaN."""
    for c in ['Team','Driver','Stint','Compound','FreshTyre']:
        if d[c].nunique(dropna=False)!=1:raise ValueError('One named contribution per endpoint: '+c)
    w=np.asarray(d['_weight'] if '_weight' in d else np.ones(len(d)),float);w=w/w.sum()
    return dict(team=str(d.Team.iloc[0]),driver=str(d.Driver.iloc[0]),stint=float(d.Stint.iloc[0]),compound=str(d.Compound.iloc[0]),fresh=bool(d.FreshTyre.iloc[0]),
                lap=float(w@d.LapNumber.to_numpy(float)),life=float(w@d.TyreLife.to_numpy(float)),age=float(w@d.TyreAgeLaps.to_numpy(float)),
                n=len(d),rows=tuple(map(int,d.row_id)) if 'row_id' in d else (),weights=tuple(map(float,w)))

def transfer_class(a,b):
    if a['driver']==b['driver'] and a['stint']==b['stint']:return 'same_stint'
    if a['driver']==b['driver']:return 'same_driver_new_stint'
    return 'cross_driver'

def class_key(a,b):
    return (transfer_class(a,b),a['compound'],b['compound'],a['fresh'],b['fresh'])

class ExactTransfer:
    """Transfer donors matched to exact endpoint race-lap schedules.

    Every endpoint target row gets one same-compound/Fresh donor row on the exact
    race lap, physical life within the fixed radius. Primary predictions require
    complete endpoint support: donor and target averages use identical weights.
    Same donor rows in source and target are allowed only for target-shared laps;
    if target supports differ they cannot be represented by one identical donor
    support. No target-team outcomes are used under any query.
    """
    def __init__(self,training_data,calibration_path=None):
        self.training_data=training_data.copy()
        if 'row_id' not in self.training_data:self.training_data['row_id']=self.training_data.index.astype(int)
        self.training_data['_clock_seconds']=pd.to_timedelta(self.training_data.LapStartTime).dt.total_seconds() if 'LapStartTime' in self.training_data else np.nan
        self.groups=[];self._endpoint_cache={};self._cache={}
        for key,g in self.training_data.groupby(['Team','Driver','Stint','Compound','FreshTyre'],sort=True):
            if len(g)<4:continue
            self.groups.append((key,g.set_index('LapNumber',drop=False)))
        self.calibration={}
        if calibration_path and Path(calibration_path).exists():self.calibration=json.loads(Path(calibration_path).read_text())

    def _matches(self,support,radius,excluded_rows,full_support):
        # Under full support, each named donor stint has at most one row at an
        # exact race lap. Removing any required row rejects that whole donor,
        # so cache the complete matches once and filter held rows afterwards.
        # Partial support is different (weights may change), retain old path.
        if full_support and excluded_rows:
            all_matches=self._matches(support,radius,set(),True)
            return [m for m in all_matches if not(set(m['rows'])&excluded_rows)]
        support_clock=pd.to_timedelta(support.LapStartTime).dt.total_seconds().to_numpy() if 'LapStartTime' in support else np.full(len(support),np.nan)
        s=describe_support(support);signature=(s['rows'],s['weights'],tuple(map(float,support.LapNumber)),tuple(map(float,support.TyreLife)),tuple(map(float,support_clock)),s['compound'],s['fresh'],radius,tuple(sorted(excluded_rows)),full_support)
        if signature in self._endpoint_cache:return self._endpoint_cache[signature]
        weights=np.array(s['weights']);matches=[]
        for key,g in self.groups:
            team,driver,stint,compound,fresh=key
            if compound!=s['compound'] or bool(fresh)!=s['fresh']:continue
            pos=[];donor=[]
            for k,row in enumerate(support.itertuples()):
                if row.LapNumber not in g.index:continue
                z=g.loc[row.LapNumber]
                if int(z.row_id) in excluded_rows or abs(float(z.TyreLife-row.TyreLife))>radius:continue
                pos.append(k);donor.append(z)
            if len(pos)<4 or (full_support and len(pos)!=len(support)):continue
            if not full_support and len(pos)/len(support)<.75:continue
            w=weights[pos];w=w/w.sum();dd=pd.DataFrame(donor)
            clock_gaps=np.abs(dd['_clock_seconds'].to_numpy()-support_clock[pos])
            matches.append(dict(team=team,driver=driver,stint=float(stint),compound=compound,fresh=bool(fresh),
                                y=float(w@dd.LapTimeSeconds.to_numpy()),rows=tuple(map(int,dd.row_id)),weights=tuple(map(float,w)),
                                target_positions=tuple(pos),coverage=len(pos)/len(support),
                                clock_mismatch_mean=float(w@clock_gaps),clock_mismatch_max=float(np.max(clock_gaps)),
                                life_mismatch=float(w@np.abs(dd.TyreLife.to_numpy()-support.iloc[pos].TyreLife.to_numpy()))))
        self._endpoint_cache[signature]=matches;return matches

    def predict(self,source_support,target_support,method='local_mean',exclude_teams=(),exclude_rows=(),strict=False,return_donors=False,full_support=True):
        if method not in ['local_mean','local_half','zero']:raise ValueError(method)
        a=describe_support(source_support);b=describe_support(target_support)
        if a['team']!=b['team']:raise ValueError('Within-team junction only')
        blocked=set(exclude_teams)|{a['team']};blocked_rows=set(map(int,exclude_rows));radius=2 if strict else 4
        key=class_key(a,b);candidates=[]
        aa=self._matches(source_support,radius,blocked_rows,full_support)
        bb=self._matches(target_support,radius,blocked_rows,full_support)
        same_target=(a['rows']==b['rows'] and a['weights']==b['weights'])
        for x in aa:
            if x['team'] in blocked:continue
            for y in bb:
                if x['team']!=y['team'] or class_key(x,y)!=key:continue
                if x['rows']==y['rows'] and x['weights']==y['weights'] and not same_target:continue
                # Comparability cannot be rescued by splitting one target named
                # driver into two donor drivers or collapsing target stint changes.
                delta=y['y']-x['y'];weight=1/(1+x['life_mismatch']+y['life_mismatch'])
                candidates.append(dict(team=x['team'],delta=float(delta),weight=float(weight),source_rows=list(x['rows']),target_rows=list(y['rows']),
                                       source_weights=list(x['weights']),target_weights=list(y['weights']),
                                       source_driver=x['driver'],target_driver=y['driver'],source_stint=x['stint'],target_stint=y['stint'],
                                       source_clock_mismatch_mean=x['clock_mismatch_mean'],source_clock_mismatch_max=x['clock_mismatch_max'],
                                       target_clock_mismatch_mean=y['clock_mismatch_mean'],target_clock_mismatch_max=y['clock_mismatch_max'],
                                       source_life_mismatch=x['life_mismatch'],target_life_mismatch=y['life_mismatch'],source_coverage=x['coverage'],target_coverage=y['coverage']))
        team_values={};team_weights={}
        for p in candidates:
            team_values[p['team']]=team_values.get(p['team'],0)+p['weight']*p['delta']
            team_weights[p['team']]=team_weights.get(p['team'],0)+p['weight']
        values={t:team_values[t]/team_weights[t] for t in team_values};nt=len(values)
        prediction=float(np.mean(list(values.values()))) if nt else np.nan
        if method=='local_half':prediction*=.5
        if method=='zero':prediction=0.
        cls=key[0]+('_compound_switch' if a['compound']!=b['compound'] else '_same_compound')
        q=self.calibration.get(method,{}).get(cls,self.calibration.get(method,{}).get('all',None))
        result=dict(prediction=prediction,n_donor_teams=nt,n_donor_shifts=len(candidates),donor_teams=sorted(values),donor_team_values=values,
                    support_grade=('supported' if nt>=2 and full_support else 'one_team' if nt else 'unsupported'),
                    transfer_class=cls,raw_class=key[0],donor_sd=float(np.std(list(values.values()),ddof=1)) if nt>=2 else np.nan,
                    empirical_abs_error_p90=q,lap_transfer=b['lap']-a['lap'],life_transfer=b['life']-a['life'],
                    compound_switch=a['compound']!=b['compound'],fresh_switch=a['fresh']!=b['fresh'],
                    driver_switch=a['driver']!=b['driver'],stint_switch=(a['driver'],a['stint'])!=(b['driver'],b['stint']),
                    maximum_donor_lap_mismatch=0.,maximum_donor_life_mismatch=max([max(p['source_life_mismatch'],p['target_life_mismatch']) for p in candidates],default=np.nan),
                    maximum_donor_clock_mismatch=max([max(p['source_clock_mismatch_max'],p['target_clock_mismatch_max']) for p in candidates],default=np.nan),
                    full_target_support=full_support,minimum_endpoint_coverage=min([min(p['source_coverage'],p['target_coverage']) for p in candidates],default=np.nan))
        if return_donors:result['donors']=candidates
        return result
