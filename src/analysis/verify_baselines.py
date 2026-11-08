"""Independently rebuild saved curves from the captured daily observations."""
from pathlib import PureWindowsPath
import json
import numpy as np
import pandas as pd

def verify_baselines(folder):
    audit=json.loads((folder/'combined_losses_delays_audit_dynamic_rules_legend_grouped_distinct_bands_no_texture_lighter2.json').read_text())
    paths=[]
    for old in audit['input_hashes']:
        parts=PureWindowsPath(old).parts
        paths.append(folder/'inputs'/('/'.join(parts[parts.index('Research_RoadRestoration')+1:])))
    daily={}
    for path in paths:
        rel=str(path).replace('\\','/')
        for match,method in [('rule_baselines/flow/','flow'),('dynamic_rules/accessibility/','accessibility'),('ga_saa10_flow_insert_seed42/','GA'),('stage2/selected_test/','RL')]:
            if match in rel:daily[method]=pd.read_csv(path)
    abc=pd.concat([pd.read_csv(p) for p in paths if p.name=='daily_abc.csv'])
    checked=0
    for curve in audit['curves']:
        if curve['method'] not in daily:continue
        frame=daily[curve['method']] if curve['metric'] in ('time_loss','flow_loss') else abc[abc.method==curve['method']]
        matrix=frame.pivot(index='day',columns='scenario',values=curve['metric']).reindex(index=curve['days'],columns=audit['scenarios']).ffill().to_numpy()
        assert matrix.shape[1]==50 and np.isfinite(matrix).all()
        for key,value in [('mean',matrix.mean(axis=1)),('p10',np.quantile(matrix,.1,axis=1)),('p90',np.quantile(matrix,.9,axis=1))]:
            np.testing.assert_allclose(value,curve[key],rtol=0,atol=1e-10)
        checked+=1
    assert checked==16
    return dict(methods=4,scenarios=50,curves_reaggregated=checked,percentiles=[10,90])
