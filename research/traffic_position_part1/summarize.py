"""Extract the deliberately small correctness casebook from reproduction outputs."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

p=argparse.ArgumentParser();p.add_argument('output',type=Path);args=p.parse_args();root=args.output
results=json.loads((root/'results.json').read_text())
summary={}
r12=pd.read_csv(root/'r12_samples.csv');L=results['r12']['reference_length_m']
rear=(r12.Phase-r12.DistanceBehindM)%L
summary['r12_wrap']={'target_wraps':int((r12.Phase.diff()<-L/2).sum()),
    'pair_straddles_coordinate_zero':int((abs(r12.Phase-rear)>L/2).sum()),
    'rear_candidates':r12.DriverBehind.value_counts().to_dict(),
    'unavailable_field_members':r12.UnavailableDrivers.value_counts().to_dict()}
direct=pd.read_csv(root/'r12_direct_ownership.csv').dropna(subset=['DirectTimeDeltaToDriverAhead'])
summary['ownership']=[]
for lap in range(9,13):
    old=direct[direct.LapNumber.eq(lap)]; new=direct[direct.DirectRearTargetLapNumber.eq(lap)]
    summary['ownership'].append(dict(target_lap=lap,
        moved_out=int((~old.DirectRearTargetLapNumber.eq(lap)).sum()),
        moved_in=int((~new.LapNumber.eq(lap)).sum()),
        contributors=new.Driver.value_counts().to_dict(),mean=float(new.DirectTimeDeltaToDriverAhead.mean())))
s=pd.read_csv(root/'r13_samples.csv');timing=pd.read_csv(root/'r13_timing.csv')
summary['r13_order_change']=[]
for clock in ['2026-09-06T13:45:43Z','2026-09-06T13:45:49Z']:
    t=pd.Timestamp(clock).timestamp();rows={}
    for driver in ['PIA','NOR']:
        group=s[s.Driver.eq(driver)];row=group.iloc[np.argmin(abs(group.SessionTime-t))]
        independent=timing[timing.Driver.eq(driver)&timing.SessionTime.eq(row.SessionTime)].iloc[0]
        rows[driver]=dict(sample_utc=pd.to_datetime(row.SessionTime,unit='s',utc=True).isoformat(),
            phase_m=row.Phase,rank=int(independent.Rank),completed_counter=int(independent.Counter),
            in_pit=bool(independent.InPit),nearest_ahead=row.DriverAhead,nearest_rear=row.DriverBehind)
    rows['NOR_minus_PIA_m']=float(rows['NOR']['phase_m']-rows['PIA']['phase_m'])
    summary['r13_order_change'].append(rows)
(root/'casebook.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
