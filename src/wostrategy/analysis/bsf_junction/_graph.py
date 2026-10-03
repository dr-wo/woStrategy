"""Frozen exact-support graph and canonical projection; no new solver."""
import hashlib, heapq, itertools
from collections import defaultdict
import numpy as np
import pandas as pd
from .constants import TEAMS
def retain(library, excluded=(), drop_pair=None, level=None):
    """No rematching after the covariate freeze; purge shared raw outcomes."""
    excluded=set(map(int,excluded));out=[]
    for old in library:
        if level is not None and old['level']>level:continue
        if drop_pair is not None and {old['a_team'],old['b_team']}==set(drop_pair):continue
        pairs=[p for p in old['pairs'] if not (int(p[0]) in excluded or int(p[1]) in excluded)]
        if len(pairs)<4:continue
        e=old.copy();e['pairs']=pairs;out.append(e)
    return out

def node_id(rows):
    return ','.join(map(str,sorted(set(map(int,rows)))))

def description(d,rows):
    rows=tuple(sorted(set(map(int,rows))))
    z=d.iloc[list(rows)]
    for col in ['Team','Driver','Stint','Compound','FreshTyre']:
        assert z[col].astype(str).nunique()==1,(col,rows)
    a=z.iloc[0]
    return dict(id=node_id(rows),rows=rows,team=str(a.Team),driver=str(a.Driver),stint=float(a.Stint),compound=str(a.Compound),fresh=str(a.FreshTyre),n=len(rows),lap=float(z.LapNumber.mean()),life=float(z.TyreLife.mean()),age=float(z.TyreAgeLaps.mean()),clock=float(pd.to_timedelta(z.LapStartTime).dt.total_seconds().mean()) if z.LapStartTime.dtype=='object' else float(z.LapStartTime.mean()))

def burden(a,b,kind):
    return dict(observation_edges=int(kind=='observation'),transfer_edges=int(kind=='transfer'),race_lap_transfer=abs(b['lap']-a['lap']),life_transfer=abs(b['life']-a['life']),recorded_age_transfer=abs(b['age']-a['age']),compound_switches=int(a['compound']!=b['compound']),driver_switches=int(a['driver']!=b['driver']),stint_switches=int((a['driver'],a['stint'])!=(b['driver'],b['stint'])),fresh_switches=int(a['fresh']!=b['fresh']),fallback_transitions=0)

class ConditionGraph:
    def __init__(self,d,library,excluded=(),drop_pair=None,queries=(),transfer_model=None,shrink=1.0,min_donors=2):
        self.d=d;self.excluded=set(map(int,excluded));self.nodes={};self.adj=defaultdict(list);self.edges=[];self.queries=[];self.transfer_attempts=[]
        self.library=retain(library,excluded,drop_pair=drop_pair)
        for e in self.library:
            p=np.asarray(e['pairs'],int)
            a=self.add_node(p[:,0]);b=self.add_node(p[:,1]);z=d.iloc[p.ravel()]
            assert not set(p.ravel()).intersection(self.excluded)
            aa=d.iloc[p[:,0]];bb=d.iloc[p[:,1]]
            life=float(np.abs(aa.TyreLife.to_numpy()-bb.TyreLife.to_numpy()).mean());lap=float(np.abs(aa.LapNumber.to_numpy()-bb.LapNumber.to_numpy()).mean())
            def clocks(x):return pd.to_timedelta(x).dt.total_seconds().to_numpy() if x.dtype=='object' else x.to_numpy(float)
            clock=float(np.abs(clocks(aa.LapStartTime)-clocks(bb.LapStartTime)).mean())
            delta=float(bb.LapTimeSeconds.mean()-aa.LapTimeSeconds.mean())
            self.add_edge(a,b,delta,1+life/4+lap/2+clock/60,'observation',source_id=e['edge_id'],level=e['level'],raw_rows=sorted(set(p.ravel().tolist())),life_mismatch=life,lap_mismatch=lap,clock_mismatch=clock,n=len(p))
        for q in queries:self.queries.append((self.add_node(q[0]),self.add_node(q[1])))
        by=defaultdict(list)
        for k,v in self.nodes.items():by[v['team']].append(k)
        for team,ids in sorted(by.items()):
            for a,b in itertools.combinations(sorted(ids),2):
                if transfer_model is None:continue
                da=self.nodes[a];db=self.nodes[b]
                result=transfer_model.predict(da['rows'],db['rows'])
                self.transfer_attempts.append(dict(a=a,b=b,**result))
                if result['n_teams']<min_donors:continue
                raw=result.get('raw_rows',[])
                assert not set(raw).intersection(self.excluded)
                assert all(d.iloc[i].Team!=team for i in raw)
                bur=burden(da,db,'transfer')
                cost=1+bur['race_lap_transfer']/10+bur['life_transfer']/4+sum(bur[k] for k in ['compound_switches','driver_switches','stint_switches','fresh_switches'])
                self.add_edge(a,b,float(result['estimate'])*shrink,cost,'transfer',source_id='donor:'+str(len(self.edges)),donors=result,shrink=shrink,raw_rows=raw)
        self._paths={}

    def add_node(self,rows):
        key=node_id(rows)
        if key not in self.nodes:self.nodes[key]=description(self.d,rows)
        return key

    def add_edge(self,a,b,delta,cost,kind,**meta):
        e=dict(id=len(self.edges),a=a,b=b,delta=delta,cost=cost,kind=kind,**meta,**burden(self.nodes[a],self.nodes[b],kind))
        self.edges.append(e);self.adj[a].append((b,e['id'],1));self.adj[b].append((a,e['id'],-1))

    def paths(self,a):
        if a in self._paths:return self._paths[a]
        best={a:(0.,0,())};parent={};todo=[(0.,0,(),a)]
        while todo:
            cost,n,path,k=heapq.heappop(todo)
            if (cost,n,path)!=best[k]:continue
            for z,eid,sign in sorted(self.adj[k]):
                e=self.edges[eid];new=(cost+e['cost'],n+1,path+(eid,))
                if z not in best or new<best[z]:
                    best[z]=new;parent[z]=(k,eid,sign);heapq.heappush(todo,(*new,z))
        self._paths[a]=(best,parent);return best,parent

    def predict(self,a,b):
        best,parent=self.paths(a)
        if b not in best:return dict(connected=False,prediction=None)
        path=[];z=b
        while z!=a:
            k,eid,sign=parent[z];path.append((eid,sign));z=k
        path=path[::-1]
        delta=sum(self.edges[eid]['delta']*sign for eid,sign in path)
        keys=['observation_edges','transfer_edges','race_lap_transfer','life_transfer','recorded_age_transfer','compound_switches','driver_switches','stint_switches','fresh_switches','fallback_transitions']
        bur={k:sum(self.edges[eid][k] for eid,sign in path) for k in keys}
        # Driver/stint changes across teams aren't junction switches; expose both total and within-team.
        for k in ['race_lap_transfer','life_transfer','recorded_age_transfer','compound_switches','driver_switches','stint_switches','fresh_switches']:
            bur['junction_'+k]=sum(self.edges[eid][k] for eid,sign in path if self.edges[eid]['kind']=='transfer')
        tr=[self.edges[eid] for eid,sign in path if self.edges[eid]['kind']=='transfer']
        return dict(connected=True,prediction=-delta,cost=best[b][0],path=path,**bur,min_donor_teams=min([e['donors']['n_teams'] for e in tr],default=0),nonexact_observation_edges=sum(self.edges[eid]['kind']=='observation' and self.edges[eid]['level']>0 for eid,sign in path),training_raw_rows=sorted(set(i for eid,sign in path for i in self.edges[eid]['raw_rows'])))

    def components(self):
        out=[];unseen=set(self.nodes)
        while unseen:
            start=min(unseen);reach=set(self.paths(start)[0]);out.append(sorted(reach));unseen-=reach
        return out

def projected_scores(cg,forecast,base,counts,shrink=1.,reference_override=None):
    """Explicit canonical-projected consensus. Never equated with a local contrast.

    Every usable condition node pays forecast(node)->baseline(team) as an
    explicitly canonical/fallback projection. p(reference) fixes its gauge.
    Equal weights reconcile copies only AFTER their projection. Disconnected
    components retain full canonical fallback; absent teams use field-mean prior.
    """
    b=np.asarray(base,float);p=np.asarray(forecast,float)
    assert np.isfinite(b).all() and np.isfinite(p).all()
    incident=defaultdict(int)
    for e in cg.edges:
        if e['kind']=='observation':incident[e['a']]+=1;incident[e['b']]+=1
    terms=[];scores=b.copy();reference=None
    if incident:
        reference=reference_override or min(incident,key=lambda k:(-incident[k],k));assert reference in incident;pref=float(p[list(cg.nodes[reference]['rows'])].mean())
        by=defaultdict(list)
        for node in sorted(incident):
            z=cg.nodes[node];out=cg.predict(reference,node)
            if not out['connected']:continue
            # predict(ref,node)=ref-node; reconstruct node's arbitrary absolute gauge.
            diff=sum(cg.edges[eid]['delta']*sign*(shrink if cg.edges[eid]['kind']=='transfer' else 1) for eid,sign in out['path'])
            raw_value=pref+diff;i=TEAMS.index(z['team']);pnode=float(p[list(z['rows'])].mean());projection=float(b[i]-pnode)
            term=dict(team=z['team'],node=node,reference=reference,raw_graph_value=raw_value,canonical_node_prediction=pnode,canonical_reference_prediction=pref,projection_delta=projection,projected_value=raw_value+projection,projection_source='same_fold_canonical_reference_functional',projection_fallback_transitions=1,reference_anchor_fallback_transitions=1,reference_anchor_source='same_fold_canonical_predicted_reference_support_mean',reference_anchor_cancels_within_component=True,projection_interpretation='engineering extrapolation only; no separately identified physical effects',**{k:v for k,v in out.items() if k not in ['prediction','training_raw_rows']})
            terms.append(term);by[z['team']].append(term)
        for team,tt in by.items():
            scores[TEAMS.index(team)]=np.mean([t['projected_value'] for t in tt])
            for t in tt:t['aggregation_weight']=1/len(tt)
    scores-=scores.mean()
    metadata=[]
    for i,team in enumerate(TEAMS):
        tt=[t for t in terms if t['team']==team];inc=[e for e in cg.edges if e['kind']=='observation' and team in [cg.nodes[e['a']]['team'],cg.nodes[e['b']]['team']]]
        tr=[e for e in cg.edges if e['kind']=='transfer' and cg.nodes[e['a']]['team']==team]
        keys=['observation_edges','transfer_edges','race_lap_transfer','life_transfer','junction_race_lap_transfer','junction_life_transfer','compound_switches','driver_switches','stint_switches','junction_compound_switches','junction_driver_switches','junction_stint_switches']
        metadata.append(dict(team=team,score=float(scores[i]),support_class='no_clean_neutral_prior' if counts[i]==0 else 'condition_network_with_canonical_projection' if tt else 'full_canonical_fallback',canonical_score=float((b-b.mean())[i]),clean_laps=int(counts[i]),projected_support_nodes=len(tt),direct_local_edge_count=len(inc),exact_local_edge_count=sum(e['level']==0 for e in inc),available_transfer_edge_count=len(tr),fallback_dependent=True,fallback_projected_term_fraction=1.,full_canonical_fallback=not bool(tt),reference_node=reference,**{f'mean_path_{k}':float(np.mean([t[k] for t in tt])) if tt else 0. for k in keys},mean_path_fallback_transitions=2. if tt else 1.,shared_reference_anchor_fallback=bool(tt),reliability='unvalidated_team_scalar; no calibrated team CI'))
    return scores,metadata,terms
