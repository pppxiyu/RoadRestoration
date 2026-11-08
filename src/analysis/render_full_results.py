"""Render three reproducible full-response RL result groups, PNG only."""
from pathlib import Path
import sys,json,hashlib,itertools,argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
HERE=ROOT/'outputs/experiments/full_rl_verification_20261106'
INPUTS=HERE/'inputs'
OUT=Path(__import__('os').environ.get('RESULTS_OUTPUT',ROOT/'outputs/sections/full_rl_performance'));OUT.mkdir(parents=True,exist_ok=True)
SUPPORT=OUT/'support';SUPPORT.mkdir(exist_ok=True)
SRC=SUPPORT/'source_data';SRC.mkdir(exist_ok=True)
from src.analysis.panel_alignment import require_matplotlib_panel_alignment
BASE=INPUTS/'.cache/babaee_horizon_pure_rl'
ROUNDS=[INPUTS/'.cache/babaee_base0005_paired_two_rounds/stage1',INPUTS/'.cache/babaee_base0005_paired_two_rounds/stage2',INPUTS/'.cache/babaee_base0005_paired_round3/stage3']
SOURCES={};STATS={}
BLUE='#4579AB';PURPLE='#7046A3';GRAY='#9EA7B1';ORANGE='#D18A42';GREEN='#328C82'
plt.rcParams.update({'font.family':'Arial','font.size':18,'axes.titlesize':21,'axes.labelsize':19,'xtick.labelsize':16,'ytick.labelsize':16,'legend.fontsize':15,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':1,'lines.linewidth':2.6,'axes.titlepad':18,'savefig.facecolor':'white','figure.facecolor':'white'})
def record(p):SOURCES[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest();return p
def js(p):return json.loads(record(p).read_text(encoding='utf8'))
def csv(p):return pd.read_csv(record(p))
def tidy(ax):ax.grid(axis='y',color='#E3E6EB',lw=.8);ax.set_axisbelow(True)
def export(fig,name):
    fig.canvas.draw()
    if len(fig.axes)>1:require_matplotlib_panel_alignment(fig,json_out=SUPPORT/f'{name}.alignment.json',strict=True)
    fig.savefig(OUT/f'{name}.png',dpi=300,bbox_inches='tight',pad_inches=.16)
    plt.close(fig)
def training():
    d=csv(BASE/'training/log/training.csv');sel=js(BASE/'selection.json');validation=d.dropna(subset=['validation_objective'])
    assert d.episode.tolist()==list(range(401));assert int(sel['selected_episode'])==150
    d.to_csv(SRC/'training_log.csv',index=False)
    y=[sel['validation_objective']]+[js(p/'selection.json')['validation_objective'] for p in ROUNDS]
    assert np.argmin(y)==2
    pd.DataFrame({'round':range(4),'validation_loss':y,'best_so_far':np.minimum.accumulate(y)}).to_csv(SRC/'outer_rounds.csv',index=False)
    fig,axes=plt.subplots(1,3,figsize=(22.5,6.3));fig.subplots_adjust(left=.055,right=.985,bottom=.2,top=.84,wspace=.32)
    a=axes[0]
    a.plot(d.episode,d.training_objective,color=BLUE,alpha=.23,lw=1,label='Recorded episodes')
    a.plot(d.episode,d.training_objective.rolling(20,min_periods=1).mean(),color=BLUE,label='20-episode moving mean')
    a.set(title='Base reinforcement learning',xlabel='Training episode',ylabel='Mean training system loss',xlim=(0,400));a.legend(frameon=False)
    a=axes[1]
    a.plot(validation.episode,validation.validation_objective,'o-',color=BLUE,ms=5,label='Validation checkpoints')
    a.plot(validation.episode,np.minimum.accumulate(validation.validation_objective),'--',color='#303A45',lw=1.7,label='Best so far')
    a.scatter([sel['selected_episode']],[sel['validation_objective']],marker='D',s=85,facecolor='white',edgecolor=PURPLE,lw=2,zorder=6,label='Selected base model')
    a.set(title='Base-policy validation',xlabel='Training episode',ylabel='Mean validation system loss',xlim=(0,400));a.legend(frameon=False)
    a=axes[2]
    a.plot(range(4),y,'o-',color=PURPLE,ms=7,label='Round candidate')
    a.plot(range(4),np.minimum.accumulate(y),'--',color='#303A45',lw=1.7,label='Best so far')
    a.scatter([2],[y[2]],marker='D',s=100,facecolor='white',edgecolor=PURPLE,lw=2,zorder=6,label='Final selected model')
    a.set(title='Iterative policy improvement',xlabel='Improvement round',ylabel='Mean validation system loss',xticks=range(4),xticklabels=['0\nBase','1','2\nSelected','3\nRejected']);a.legend(frameon=False,loc='upper right')
    for a in axes:tidy(a)
    export(fig,'01_training_process')
    STATS['training']=dict(episodes=401,validation_worlds=11,validation_checkpoints=len(validation),episodes_without_validation=int(d.validation_objective.isna().sum()),seed_count=1,training_metric='Recorded batch-average physical system loss, NOT cumulative horizon reward',smoothing='Trailing 20-episode mean, min_periods=1, only training panel',selected_base_episode=150,round_losses=y,selected_round=2,base_to_final_reduction_percent=100*(y[0]-y[2])/y[0],round4='Preserved historical extra round excluded under recorded retrospective first-nonimprovement stopping rule')
def recovery():
    ex=INPUTS/'outputs/experiments/n11_poi_hidden6_discovery_response7d_1day_lhs50_s42_0514ff84/data/methods/ga_saa10_flow_insert_seed42'
    old=INPUTS/'.cache/results_figures_20261021'
    paths=[('Static flow priority',old/'rule_baselines/flow/test_daily.csv',BLUE,'--'),('Dynamic accessibility / duration',old/'dynamic_rules/accessibility/test_daily.csv',ORANGE,'-.'),('GA',ex/'results/test_daily.csv',GREEN,':'),('Full-environment RL',ROUNDS[1]/'selected_test/results/test_daily.csv',PURPLE,'-')]
    frames=[csv(p) for _,p,_,_ in paths];ids=[f'test_{i:03d}' for i in range(50)]
    assert all(sorted(d.scenario.unique())==ids for d in frames)
    horizon=max(int(d.day.max()) for d in frames);days=np.arange(1,horizon+1);rows=[];allraw=[]
    fig,ax=plt.subplots(figsize=(12.8,6.5));fig.subplots_adjust(left=.10,right=.98,bottom=.16,top=.80)
    for (name,_,color,ls),d in zip(paths,frames):
        assert not d.duplicated(['scenario','day']).any()
        v=d.pivot(index='day',columns='scenario',values='total_flow_fraction').reindex(index=days,columns=ids)
        assert v.iloc[0].notna().all()
        filled=v.ffill();assert filled.notna().all().all()
        # Ended episodes have reached their recorded restoration criterion.
        # Carry their last observation forward, never replace by invented 100%.
        means=100*filled.mean(axis=1).to_numpy()
        ax.plot(days,means,color=color,ls=ls,lw=3.4 if 'RL' in name else 2.5,label=name)
        for j,day in enumerate(days):rows.append(dict(method=name,day=int(day),mean_realized_trip_percent=float(means[j]),observed_scenarios=int(v.loc[day].notna().sum()),total_scenarios=50))
        raw=filled.rename_axis(index='day',columns='scenario').stack().rename('realized_trip_fraction').reset_index();raw['method']=name;allraw.append(raw)
    ax.set(xlabel='Days after disaster',ylabel='Realized trips / normal demand (%)',xlim=(1,horizon),ylim=(0,102));tidy(ax)
    fig.legend(*ax.get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.54,.99),ncol=2,frameon=False,fontsize=16)
    pd.DataFrame(rows).to_csv(SRC/'recovery_means.csv',index=False);pd.concat(allraw).to_csv(SRC/'recovery_scenario_values.csv',index=False)
    export(fig,'03_recovery_trajectory')
    STATS['recovery']=dict(scenarios=50,horizon=horizon,metric='Actual served trips / normal total OD demand; traffic service recovery, not average bridge physical functionality',aggregation='Equal-weight mean over same 50 test scenarios',after_episode_end='Last observed restored fraction carried forward; not newly simulated',uncertainty='Mean trajectories only; no invented training-seed uncertainty')
def actions():
    js(HERE/'action_complete.json');contract=js(HERE/'action_contract.json')
    names=['Base RL','Round 1','Round 2 (selected)','Round 3 (rejected)'];pairs=[];state_acc=[];top_rows=[]
    for i in range(50):
        d=js(HERE/'action_results'/f'test_{i:03d}.json');cost=np.array([b['remaining_loss'] for b in d['branches']]);qs=np.array(d['q_values'])
        a,b=np.triu_indices(len(cost),1);actual=cost[b]-cost[a];eligible=abs(actual)>1e-8
        # The simulation ranking is shared by all networks. Include cutoff ties.
        top_masks={}
        for n in [3,6,9]:
            cutoff=np.sort(cost)[min(n,len(cost))-1]
            member=cost<=cutoff+1e-8
            top_masks[n]=((member[a]&member[b])&eligible,member)
        assert np.array_equal(top_masks[9][0],eligible)
        acc=[]
        for k,name in enumerate(names):
            pred=qs[k,a]-qs[k,b]
            acc.append(float(np.mean((pred[eligible]*actual[eligible])>0)))
            for n in [3,6,9]:
                mask,member=top_masks[n]
                assert mask.any()
                correct=int(np.count_nonzero(pred[mask]*actual[mask]>0))
                top_rows.append(dict(scenario=d['scenario'],model=name,round=k,top_x=n,
                    candidates=len(cost),top_action_count=int(member.sum()),
                    eligible_pairs=int(mask.sum()),correct_pairs=correct,
                    accuracy=correct/int(mask.sum())))
            for j in range(len(a)):pairs.append(dict(scenario=d['scenario'],model=name,round=k,road_a=d['roads'][a[j]],road_b=d['roads'][b[j]],actual_advantage=float(actual[j]),predicted_advantage=float(pred[j]),eligible_for_accuracy=bool(eligible[j]),**{f'within_top_{n}':bool(top_masks[n][0][j]) for n in [3,6,9]}))
        state_acc.append(acc)
    pair=pd.DataFrame(pairs);pair.to_csv(SRC/'heldout_action_pairs.csv',index=False)
    values=np.array(state_acc);pd.DataFrame(values,index=[f'test_{i:03d}' for i in range(50)],columns=names).to_csv(SRC/'state_pair_accuracy.csv',index_label='scenario')
    means=100*values.mean(axis=0);rng=np.random.default_rng(20261107)
    boot=100*values[rng.integers(0,50,(2000,50))].mean(axis=1);lo,hi=np.quantile(boot,[.025,.975],axis=0)
    top=pd.DataFrame(top_rows);top.to_csv(SRC/'topx_pair_state_accuracy.csv',index=False)
    np.testing.assert_allclose(top[top.top_x==9].pivot(index='scenario',columns='round',values='accuracy').reindex(columns=range(4)).to_numpy(),values)
    top_summary=[]
    for n in [3,6,9]:
        t=top[top.top_x==n].pivot(index='scenario',columns='round',values='accuracy').reindex(columns=range(4)).to_numpy()
        draws=np.random.default_rng(20261107).integers(0,50,(2000,50))
        intervals=np.quantile(100*t[draws].mean(axis=1),[.025,.975],axis=0)
        for k,name in enumerate(names):
            top_summary.append(dict(model=name,round=k,top_x=n,accuracy_percent=float(100*t[:,k].mean()),
                total_pairs=int(top[(top.top_x==n)&(top['round']==k)].eligible_pairs.sum()),states=50,
                bootstrap_lower=float(intervals[0,k]),bootstrap_upper=float(intervals[1,k])))
    ts=pd.DataFrame(top_summary);ts.to_csv(SRC/'topx_pair_summary.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(18,8.8));fig.subplots_adjust(left=.075,right=.985,bottom=.18,top=.89,wspace=.30)
    for ax in axes:ax.set_box_aspect(1)
    ax=axes[0]
    for name,color in [('Base RL',BLUE),('Round 1',ORANGE),('Round 2 (selected)',PURPLE)]:
        d=pair[pair.model==name];ax.scatter(d.actual_advantage,d.predicted_advantage,s=18,alpha=.60,color=color,edgecolors='none',label=name)
    displayed=pair[pair['round'].isin([0,1,2])];low=-13.2;high=6.2;pad=0
    # Shared broken interval contains no x or y observation from any displayed model.
    break_lo,break_hi,gap=-9.6,-6.4,.35
    def forward(v):
        v=np.asarray(v);return np.where(v<break_lo,v,np.where(v<=break_hi,break_lo+(v-break_lo)*gap/(break_hi-break_lo),v-(break_hi-break_lo-gap)))
    def inverse(v):
        v=np.asarray(v);return np.where(v<break_lo,v,np.where(v<=break_lo+gap,break_lo+(v-break_lo)*(break_hi-break_lo)/gap,v+(break_hi-break_lo-gap)))
    observations=np.r_[displayed.actual_advantage,displayed.predicted_advantage]
    assert not ((observations>break_lo)&(observations<break_hi)).any()
    ax.set_xscale('function',functions=(forward,inverse));ax.set_yscale('function',functions=(forward,inverse))
    crop_counts={name:int(((d.actual_advantage<low)|(d.actual_advantage>high)|(d.predicted_advantage<low)|(d.predicted_advantage>high)).sum()) for name,d in displayed.groupby('model')}
    (SUPPORT/'action_plot_crop.json').write_text(json.dumps(dict(limits=[low,high],outside_view_by_model=crop_counts,all_points_retained_in_csv=True,statistics_use_all_points=True),indent=2))
    trends=[]
    for name,color in [('Base RL',BLUE),('Round 1',ORANGE),('Round 2 (selected)',PURPLE)]:
        d=displayed[displayed.model==name]
        slope,intercept=np.polyfit(d.actual_advantage,d.predicted_advantage,1)
        trends.append(dict(model=name,slope=float(slope),intercept=float(intercept),points=len(d),method='Descriptive ordinary least squares on all action pairs; no independence inference'))
        for start,end in [(break_hi,high)]:
            xx=np.linspace(start,end,200);yy=slope*xx+intercept
            yy=np.where((yy>break_lo)&(yy<break_hi),np.nan,yy)
            ax.plot(xx,yy,color=color,lw=2.4,alpha=.75,zorder=5)
    for index,(start,end) in enumerate([(low,break_lo),(break_hi,high)]):
        ax.plot([start,end],[start,end],color='#444B55',ls='--',lw=1.5,label='y = x' if index==0 else None)
    ax.axhline(0,color='#A8AFB7',lw=.8);ax.axvline(0,color='#A8AFB7',lw=.8)
    ax.set(title='Action-value differences',xlabel='Simulated remaining-loss advantage',ylabel='Predicted Q-value difference',xlim=(low-pad,high+pad),ylim=(low-pad,high+pad));ax.set_aspect('equal',adjustable='box');ax.legend(frameon=False,fontsize=14,loc='upper left',markerscale=1.6)
    ticks=[-12,-10,-6,-4,-2,0,2,4,6];ax.set_xticks(ticks);ax.set_yticks(ticks)
    # Visible double slashes on both axes mark the same omitted empty interval.
    frac=float((forward((break_lo+break_hi)/2)-forward(low))/(forward(high)-forward(low)))
    for offset in [-.004,.004]:
        ax.plot([frac+offset-.004,frac+offset+.004],[-.009,.009],transform=ax.transAxes,color='white',lw=5,clip_on=False,zorder=9)
        ax.plot([frac+offset-.004,frac+offset+.004],[-.009,.009],transform=ax.transAxes,color='#333333',lw=1.4,clip_on=False,zorder=10)
        ax.plot([-.009,.009],[frac+offset-.004,frac+offset+.004],transform=ax.transAxes,color='white',lw=5,clip_on=False,zorder=9)
        ax.plot([-.009,.009],[frac+offset-.004,frac+offset+.004],transform=ax.transAxes,color='#333333',lw=1.4,clip_on=False,zorder=10)
    ax=axes[1];ax.plot(range(3),means[:3],'o-',color=PURPLE,ms=8,label='All-pair ordering accuracy')
    ax.fill_between(range(3),lo[:3],hi[:3],color=PURPLE,alpha=.15)
    for n,color,marker,ls in [(6,GREEN,'s','--'),(3,BLUE,'^','-.')]:
        rates=ts[(ts.top_x==n)&(ts['round']<3)].sort_values('round').accuracy_percent.to_numpy()
        intervals=ts[(ts.top_x==n)&(ts['round']<3)].sort_values('round')
        ax.fill_between(range(3),intervals.bootstrap_lower.to_numpy(),intervals.bootstrap_upper.to_numpy(),color=color,alpha=.15)
        label=f'Pairs within top {n}'
        ax.plot(range(3),rates,color=color,marker=marker,ls=ls,ms=8,label=label)
    ax.fill_between([],[],[],color=GRAY,alpha=.20,label='Shading: 95% bootstrap CI')
    ax.set(title='Action-ordering accuracy',xlabel='Improvement round',ylabel='Correctly ordered action pairs (%)',xticks=range(3),xticklabels=['0\nBase','1','2\nSelected'],ylim=(0,100));ax.legend(frameon=False,fontsize=14,loc='lower left');tidy(ax)
    fig.canvas.draw()
    box=axes[0].get_window_extent();p=axes[0].transData.transform([[0,0],[1,1]])
    angle=float(np.degrees(np.arctan2(*(p[1]-p[0])[::-1])))
    assert abs(box.width-box.height)<1e-6 and abs(angle-45)<1e-8
    assert not any(crop_counts.values())
    (SUPPORT/'action_plot_geometry.json').write_text(json.dumps(dict(left_width_pixels=box.width,left_height_pixels=box.height,y_equals_x_angle_degrees=angle,displayed_rounds=[0,1,2],broken_interval=[break_lo,break_hi],display_gap=gap,all_points_visible=True),indent=2),encoding='utf8')
    (SUPPORT/'action_trend_lines.json').write_text(json.dumps(trends,indent=2),encoding='utf8')
    export(fig,'02_action_value_verification')
    STATS['actions']=dict(states=50,decision_index=2,unique_action_pairs=int(len(pair)/4),branches=sum(len(js(HERE/'action_results'/f'test_{i:03d}.json')['roads']) for i in range(50)),models=names,accuracy_percent=means.tolist(),bootstrap_lower=lo.tolist(),bootstrap_upper=hi.tolist(),pair_weighting='Average fraction of correctly ranked non-tied pairs within each state, then equal-weight mean over 50 states',contract=contract)
    STATS['actions'].update(displayed_rounds=[0,1,2],topx_pair_definition='Retain each unordered pair once only if both actions are in the simulation top X. Include all cost ties within 1e-8 of cutoff, exclude pairs with abs(cost difference)<=1e-8. Same pairs for all models. Compute accuracy within state then equal-weight mean over 50 states. Top9 equals all pairs, not a hit rate. Top1 has no eligible pair and is omitted.',topx_pair_summary=top_summary,scatter_limits=[low,high],scatter_alpha=.60,trend_alpha=.75,trend_visible_x_range=[break_hi,high],outside_view_by_model=crop_counts,axis_break=[break_lo,break_hi],trend_lines=trends,confidence_bands='All three curves, same 2000 scenario bootstrap draws; pointwise 95% intervals')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--without-actions',action='store_true');a=ap.parse_args()
    training();recovery()
    if not a.without_actions:actions()
    for p,h in SOURCES.items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h
    (SUPPORT/'plot_audit.json').write_text(json.dumps(dict(statistics=STATS,source_sha256=SOURCES,png_only=True),indent=2),encoding='utf8')
    print(json.dumps(STATS,indent=2))
if __name__=='__main__':main()
