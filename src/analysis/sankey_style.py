"""Render the two prespecified PNGs from verified returned data only."""
from pathlib import Path
import csv,json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch,Rectangle,Patch

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'returned/result/report'
OUTPUT=ROOT/'outputs'
STYLE={'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
       'font.size':18,'axes.labelsize':20,'xtick.labelsize':17,'ytick.labelsize':17,
       'legend.fontsize':15,'axes.spines.top':False,'axes.spines.right':False,
       'figure.facecolor':'white','axes.facecolor':'white'}

def sankey():
    counts=np.zeros((5,6),int)
    with (DATA/'sankey.csv').open() as f:
        for r in csv.DictReader(f):counts[int(r['simplified_rank'])-1,int(r['response_rank_bucket'])-1]=int(r['actions'])
    assert counts.sum()>0,'No unambiguous states available for a Sankey; do not fabricate flows'
    scale=.76/counts.sum();gap=.025
    def positions(values):
        height=values*scale;top=(1+height.sum()+gap*(len(values)-1))/2;out=[]
        for h in height:out.append((top-h,top));top-=h+gap
        return out
    left=positions(counts.sum(axis=1));right=positions(counts.sum(axis=0))
    lp=np.array([p[1] for p in left]);rp=np.array([p[1] for p in right])
    flows=[]
    for i in range(5):
        for j in range(6):
            height=counts[i,j]*scale
            if height:
                flows.append((i,j,lp[i]-height,lp[i],rp[j]-height,rp[j]));lp[i]-=height;rp[j]-=height
    fig,ax=plt.subplots(figsize=(12.4,6.8))
    for i,j,yl0,yl1,yr0,yr1 in sorted(flows,key=lambda f:(f[0]==0 or f[1]==0)):
        color='#4477AA' if i==0 else '#CC8844' if j==0 else '#B9BEC5'
        verts=[(.25,yl0),(.43,yl0),(.57,yr0),(.75,yr0),(.75,yr1),(.57,yr1),(.43,yl1),(.25,yl1),(.25,yl0)]
        codes=[MPath.MOVETO,MPath.CURVE4,MPath.CURVE4,MPath.CURVE4,MPath.LINETO,MPath.CURVE4,MPath.CURVE4,MPath.CURVE4,MPath.CLOSEPOLY]
        ax.add_patch(PathPatch(MPath(verts,codes),facecolor=color,edgecolor='none',alpha=.7 if i==0 or j==0 else .35))
    # Percentages use each flow's source rank as the denominator.
    for i,j,yl0,yl1,yr0,yr1 in flows:
        if i!=0 and j!=0:continue
        t=.77 if i==0 else .23
        blend=3*t*t-2*t*t*t
        x=(1-t)**3*.25+3*(1-t)**2*t*.43+3*(1-t)*t*t*.57+t**3*.75
        y=(1-blend)*(yl0+yl1)/2+blend*(yr0+yr1)/2
        ax.text(x,y,f'{100*counts[i,j]/counts[i].sum():.1f}%',
                ha='center',va='center',fontsize=16,
                color='#245581' if i==0 else '#925318',
                bbox=dict(boxstyle='round,pad=.12',facecolor='white',edgecolor='none',alpha=.93),zorder=5)
    for side,positions_,x in [('left',left,.225),('right',right,.75)]:
        for k,(low,high) in enumerate(positions_):
            ax.add_patch(Rectangle((x,low),.025,high-low,facecolor='#4477AA' if k==0 else '#737C86',edgecolor='white',linewidth=.7))
            label=f'Rank {k+1}' if k<5 else 'Rank 6 or below'
            ax.text(x-.025 if side=='left' else x+.05,(low+high)/2,label,ha='right' if side=='left' else 'left',va='center',fontsize=18)
    ax.text(.235,1.035,'Fixed demand path',ha='center',va='bottom',fontsize=20)
    ax.text(.765,1.035,'Demand response',ha='center',va='bottom',fontsize=20)
    ax.legend(handles=[Patch(facecolor='#4477AA',label='Initially ranked first'),
                       Patch(facecolor='#CC8844',label='Promoted to first'),
                       Patch(facecolor='#B9BEC5',label='Other transitions')],loc='upper center',bbox_to_anchor=(.5,-.015),ncol=3,frameon=False)
    ax.set(xlim=(-.02,1.11),ylim=(-.02,1.09));ax.axis('off')
    fig.savefig(OUTPUT/'top5_rank_transitions_percentages.png',dpi=600,bbox_inches='tight',facecolor='white');plt.close(fig)

def density():
    rows=json.loads((DATA/'state_results.json').read_text())
    low=np.sort([r['penalty_min'] for r in rows]);high=np.sort([r['penalty_max'] for r in rows]);n=len(rows)
    assert low.min()>=-1e-10 and high.min()>=-1e-10
    maximum=max(float(high.max()),1e-6)
    # A density histogram retains the mass at exactly zero without spreading
    # it over negative costs or interpreting it as a smooth continuous density.
    edges=np.arange(0,np.ceil(maximum/.25)*.25+.125,.25)
    low_counts,_=np.histogram(low,bins=edges)
    high_counts,_=np.histogram(high,bins=edges)
    low_density=low_counts/(n*np.diff(edges))
    high_density=high_counts/(n*np.diff(edges))
    assert low_counts.sum()==high_counts.sum()==n
    assert np.isclose(np.sum(low_density*np.diff(edges)),1)
    assert np.isclose(np.sum(high_density*np.diff(edges)),1)
    fig,ax=plt.subplots(figsize=(11.4,6.7),constrained_layout=True)
    if np.array_equal(low,high):
        ax.stairs(low_density,edges,color='#4477AA',lw=2.4)
    else:
        ax.stairs(low_density,edges,color='#4477AA',alpha=.12,fill=True)
        ax.stairs(low_density,edges,color='#4477AA',lw=2.5,label='Lowest cost among tied choices')
        ax.stairs(high_density,edges,color='#CC8844',lw=2.5,linestyle='--',label='Highest cost among tied choices')
        ax.legend(frameon=False,loc='upper right')
    ax.set_xlabel('Additional cumulative system loss from ignoring response')
    ax.set_ylabel('Probability density')
    ax.set_xlim(0,edges[-1]);ax.set_ylim(bottom=0)
    ax.grid(axis='y',color='#D9DDE2',lw=.8);ax.set_axisbelow(True)
    fig.savefig(OUTPUT/'decision_cost_density.png',dpi=600,bbox_inches='tight',facecolor='white');plt.close(fig)
    np.savetxt(OUTPUT/'decision_cost_density_bins.csv',np.column_stack([edges[:-1],edges[1:],low_counts,high_counts,low_density,high_density]),delimiter=',',header='bin_left,bin_right,lowest_choice_count,highest_choice_count,lowest_choice_density,highest_choice_density',comments='')

def main():
    assert json.loads((ROOT/'import_audit.json').read_text())['verified']
    OUTPUT.mkdir(exist_ok=True)
    with mpl.rc_context(STYLE):sankey()

if __name__=='__main__':main()
