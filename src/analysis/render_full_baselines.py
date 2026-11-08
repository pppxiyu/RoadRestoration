"""Subset the saved legacy figure data without running experiments."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerTuple

root = Path(__file__).resolve().parents[2]/'outputs/experiments/full_baseline_comparison_20261021'
data = json.loads((root / 'combined_losses_delays_audit_dynamic_rules_legend_grouped_distinct_bands_no_texture_lighter2.json').read_text(encoding='utf-8'))
assert len(data['scenarios']) == 50
styles = [
    ('flow', 'Static: flow priority', '#4F83B8', '--', '#3479BB', .075),
    ('accessibility', 'Dynamic: accessibility / duration', '#E39B35', (0,(8,2,2,2)), '#EC8A18', .105),
    ('GA', 'GA', '#238B83', (0,(6,3)), '#009B87', .135),
    ('RL', 'Final RL', '#6A3D9A', '-', '#843FBA', .135),
]
metrics = ['time_loss','flow_loss','A','B']
titles = ['Travel-time loss','Flow loss','Road-damage free-flow delay','Actual-network congestion delay']
labels = ['Daily travel-time loss','Daily flow loss','Mean delay on reachable\nOD pairs (min)','Mean delay on reachable\nOD pairs (min)']
plt.rcParams.update({'font.family':'Arial','font.size':18,'axes.titlesize':21,
    'axes.labelsize':19,'xtick.labelsize':16,'ytick.labelsize':16,
    'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':1.1,'axes.titlepad':16})
fig, grid = plt.subplots(2,2,figsize=(18,10.8))
fig.subplots_adjust(left=.08,right=.985,bottom=.085,top=.80,wspace=.25,hspace=.50)
handles=[]
for j,(ax,metric,title,label) in enumerate(zip(grid.ravel(),metrics,titles,labels)):
    for method,name,color,ls,fill,alpha in styles:
        matches=[c for c in data['curves'] if c['method']==method and c['metric']==metric]
        assert len(matches)==1
        c=matches[0]
        days=np.array(c['days'])
        assert np.isfinite(np.array([c['mean'],c['p10'],c['p90']])).all()
        ax.fill_between(days,c['p10'],c['p90'],color=fill,alpha=alpha,linewidth=0,zorder=1)
        for bound in ['p10','p90']:
            ax.plot(days,c[bound],color=fill,alpha=.285,lw=.65,zorder=1.2)
        line,=ax.plot(days,c['mean'],color=color,ls=ls,lw=3.1 if method=='RL' else 2.6,zorder=4 if method=='RL' else 3)
        if j==0:
            handles.append((Patch(facecolor=fill,alpha=.135,linewidth=0),line))
    ref,=ax.plot(days,data['no_road_damage_baselines'][metric],color='#50545C',ls=(0,(3,3)),lw=2,zorder=2)
    if j==0:
        reference_handle=ref
    ax.set(title=title,xlabel='Days after disaster',ylabel=label,xlim=(1,int(days[-1])),xticks=np.arange(1,int(days[-1])+1,20))
    ax.grid(axis='y',color='#E5E7EB',lw=.9)
    ax.set_axisbelow(True)
fig.legend(handles+[reference_handle,Patch(facecolor='#808080',alpha=.16,edgecolor='none')],
    [s[1] for s in styles]+['No road damage','10th–90th percentiles'],
    loc='upper center',bbox_to_anchor=(.53,.955),ncol=3,frameon=False,fontsize=16,
    handlelength=3,columnspacing=2,labelspacing=.7,handler_map={tuple:HandlerTuple(ndivide=1)})
target=Path(__import__('os').environ.get('RESULTS_OUTPUT',Path(__file__).resolve().parents[2]/'outputs/sections/full_rl_performance'))/'04_baseline_comparison.png'
target.parent.mkdir(parents=True,exist_ok=True)
from src.analysis.panel_alignment import require_matplotlib_panel_alignment
(target.parent/'support').mkdir(exist_ok=True)
require_matplotlib_panel_alignment(fig,json_out=target.parent/'support/baseline_alignment.json',strict=True)
fig.savefig(target,dpi=300,bbox_inches='tight',pad_inches=.15,facecolor='white')
plt.close(fig)
print(target)
