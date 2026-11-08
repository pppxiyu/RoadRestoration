"""Render only the two requested figures from independently verified new data."""
from pathlib import Path
import json,hashlib,csv
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.cbook import boxplot_stats
from src.analysis import sankey_style

REPO=Path(__file__).resolve().parents[2]
ROOT=REPO/'outputs/experiments/demand_response_anchor_20261103'
DATA=ROOT/'returned/result/report'
OUT=Path(__import__('os').environ.get('RESULTS_OUTPUT',REPO/'outputs/sections/behavior_effects'))
SUPPORT=OUT/'support'

def main():
    assert json.loads((ROOT/'import_audit.json').read_text())['verified']
    OUT.mkdir(parents=True,exist_ok=True);SUPPORT.mkdir(exist_ok=True)
    rows=json.loads((DATA/'state_results.json').read_text())
    summary=json.loads((DATA/'summary.json').read_text())
    values=np.array([r['relative_cost_percent'] for r in rows if r['relative_cost_percent'] is not None],float)
    assert np.isfinite(values).all() and (values>=-1e-10).all()
    positive=values[values>0]
    assert len(positive)>0,'No positive costs: do not fabricate the requested distribution'
    sankey_style.DATA=DATA;sankey_style.OUTPUT=OUT
    with mpl.rc_context(sankey_style.STYLE):
        sankey_style.sankey()
        ordered=np.sort(positive)
        gap_index=int(np.argmax(np.diff(ordered)))
        broken=bool(ordered[-1]>3*ordered[gap_index] and np.diff(ordered)[gap_index]>50)
        if broken:
            left_max=float(np.ceil(ordered[gap_index]/10)*10)
            right_min=float(np.floor(ordered[gap_index+1]/10)*10)
            right_max=float(np.ceil(ordered[-1]/10)*10)
            fig,(ax,tail)=plt.subplots(1,2,figsize=(11.4,5.5),sharey=True,
                gridspec_kw=dict(width_ratios=[5,1],wspace=.18))
            axes=[ax,tail]
            limits=[(0,left_max),(right_min,right_max)]
        else:
            fig,ax=plt.subplots(figsize=(11.4,5.5));axes=[ax]
            limits=[(0,max(5,float(np.ceil(positive.max()/5)*5)))]
        fig.subplots_adjust(left=.055,right=.985,top=.98,bottom=.24)
        jitter=np.random.default_rng(20261103).uniform(-.16,.16,len(positive))
        blue,ink='#4477AA','#283849'
        for panel,lim in zip(axes,limits):
            panel.scatter(positive,jitter,s=31,color=blue,alpha=.55,linewidths=0)
            panel.set(xlim=lim,ylim=(-.78,.4))
            panel.set_yticks([]);panel.spines['left'].set_visible(False)
            panel.grid(axis='x',color='#E0E4E8',lw=.8);panel.set_axisbelow(True)
        if broken:
            ax.set_xticks(np.arange(0,left_max+1,20));tail.set_xticks([right_min,right_max])
            for panel,x in [(ax,1),(tail,0)]:
                panel.plot([x-.009,x+.009],[-.012,.012],transform=panel.transAxes,
                    color=ink,lw=1.2,clip_on=False)
            tail.tick_params(axis='y',left=False,labelleft=False)
        ax.boxplot([positive],positions=[-.48],orientation='horizontal',widths=.17,
            showfliers=False,whis=1.5,patch_artist=True,
            boxprops=dict(facecolor='#DAE5EF',edgecolor=ink,linewidth=1.7),
            medianprops=dict(color=ink,linewidth=2.3),
            whiskerprops=dict(color=ink,linewidth=1.7),capprops=dict(color=ink,linewidth=1.7))
        ax.set_yticks([])
        stats=boxplot_stats(positive,whis=1.5)[0]
        for key,anchor_y,label_y in [('whislo',-.5225,-.66),('q1',-.395,-.31),
                ('med',-.565,-.66),('q3',-.395,-.31),('whishi',-.5225,-.66)]:
            x=float(stats[key]);label=f'{x:.4f}%' if key=='whislo' else f'{x:.2f}%'
            ax.annotate(label,xy=(x,anchor_y),xytext=(x,label_y),ha='center',va='center',fontsize=15,color=ink,
                arrowprops=dict(arrowstyle='-',color=ink,lw=.8),
                bbox=dict(facecolor='white',edgecolor='none',pad=1.5),annotation_clip=False)
        fig.supxlabel('Increase in cumulative system loss (%)',y=.125,fontsize=20)
        fig.legend(handles=[Line2D([],[],marker='o',linestyle='',color=blue,label='Individual decisions'),
            Patch(facecolor='#DAE5EF',edgecolor=ink,label='Median and interquartile range')],
            loc='lower center',bbox_to_anchor=(.5,.015),ncol=2,frameon=False,fontsize=14)
        fig.savefig(OUT/'decision_cost_positive_scatter_boxplot_labeled.png',dpi=600,bbox_inches='tight',facecolor='white')
        plt.close(fig)
    audit=dict(states=len(rows),positive_states=len(positive),zero_states=int(np.sum(values==0)),
        undefined_relative_states=len(rows)-len(values),sankey_states=summary['sankey_states'],
        sankey_tie_excluded=summary['sankey_excluded'],numerical_exclusions=summary['numerical_exclusions'],
        box_statistics={k:float(stats[k]) for k in ('whislo','q1','med','q3','whishi')},
        data_sha256=hashlib.sha256((DATA/'state_results.json').read_bytes()).hexdigest(),
        sankey_sha256=hashlib.sha256((DATA/'sankey.csv').read_bytes()).hexdigest(),
        axis_limits=limits,broken_axis=broken,maximum_positive_percent=float(positive.max()),
        note='Positive values only; all outliers retained; optimistic resolution within simplified near-best actions; 1.5 IQR whiskers')
    (SUPPORT/'ranking_figure_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    with (SUPPORT/'decision_cost_relative.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f);writer.writerow(['state_id','world','loss_increase_percent'])
        writer.writerows((r['state_id'],r['world'],r['relative_cost_percent']) for r in rows)
    print(json.dumps({k:v for k,v in audit.items() if k!='numerical_exclusions'}))

if __name__=='__main__':main()
