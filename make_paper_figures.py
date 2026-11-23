"""Recreates Figures 3-6 of the PSY412 term paper from the frozen FINAL data.
Run from the repository root: python analysis/make_paper_figures.py
"""
import pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
D='data/final/'; E='data/eye_tracking/'; Path('figures').mkdir(exist_ok=True)
INC,EXC='#2a78d6','#eb6834'; INK='#0b0b0b'; MUT='#52514e'
plt.rcParams.update({'font.family':'serif','font.size':10,'axes.edgecolor':MUT,'axes.labelcolor':INK,
 'xtick.color':MUT,'ytick.color':MUT,'axes.spines.top':False,'axes.spines.right':False,'lines.linewidth':2})
col=lambda c: INC if c=='included' else EXC
mk=lambda c: 'o' if c=='included' else 's'
q=pd.read_csv(D+'PSY412_FINAL_questionnaire_scores.csv')
b=pd.read_csv(D+'PSY412_FINAL_behaviour_participant_stage_task.csv')
s=pd.read_csv(E+'PSY412_eye_candidate_QC_filtered_sensitivity.csv')
def lab(ax,x,ys,names,colors=None,dx=0.06,gap=None):
    # place labels avoiding overlap
    order=sorted(range(len(ys)),key=lambda i:ys[i]); placed=[]
    lo,hi=ax.get_ylim(); g=gap or (hi-lo)*0.045
    for i in order:
        y=ys[i]
        if placed and y-placed[-1]<g: y=placed[-1]+g
        placed.append(y); ax.text(x+dx,y,names[i],va='center',fontsize=8.5,color=INK)
# Fig 3 manipulation check
f,ax=plt.subplots(1,2,figsize=(8,3.4))
for k,(cond,xc) in enumerate([('included',0),('excluded',1)]):
    d=q[q.condition==cond]
    import numpy as np
    a=d.dropna(subset=['cyberball_actual_pct']); e=d.dropna(subset=['estimated_throws_pct'])
    ax[0].scatter(xc-0.12+np.linspace(-0.05,0.05,len(a)),a.cyberball_actual_pct,color=col(cond),marker=mk(cond),s=45,zorder=3,edgecolor='white',linewidth=1)
    ax[0].scatter(xc+0.12+np.linspace(-0.05,0.05,len(e)),e.estimated_throws_pct,facecolor='white',edgecolor=col(cond),marker=mk(cond),s=45,zorder=3,linewidth=1.6)
    fe=d.dropna(subset=['felt_excluded'])
    import numpy as np
    jit=np.linspace(-0.1,0.1,len(fe))
    ax[1].scatter(xc+jit,fe.felt_excluded,color=col(cond),marker=mk(cond),s=45,zorder=3,edgecolor='white',linewidth=1)
ax[0].axhline(100/3,ls=':',color=MUT,lw=1); ax[0].text(1.45,35,'equal share (33%)',fontsize=8,color=MUT,ha='right')
for a_ in ax: a_.set_xticks([0,1]); a_.set_xticklabels(['Inclusion','Exclusion']); a_.set_xlim(-0.5,1.5)
ax[0].set_ylabel('% of throws received'); ax[0].set_title('Logged (filled) vs self-estimated (open)',fontsize=10)
ax[0].text(-0.12,-9,'log',ha='center',fontsize=8,color=MUT); ax[0].text(0.12,-9,'estimate',ha='center',fontsize=8,color=MUT)
ax[0].text(0.88,-9,'log',ha='center',fontsize=8,color=MUT); ax[0].text(1.12,-9,'estimate',ha='center',fontsize=8,color=MUT)
ax[0].set_ylim(-14,90)
ax[1].set_ylabel('"I felt excluded" (1–5)'); ax[1].set_ylim(0.5,5.5); ax[1].set_title('Felt exclusion',fontsize=10)
f.tight_layout(); f.savefig('figures/fig3_manipulation_check.png',dpi=200); plt.close()
# Fig 4 peripheral accuracy for P001,P004,P006
f,ax=plt.subplots(1,2,figsize=(8,3.4),sharey=True)
for a_,task,tt in [(ax[0],'pro','Prosaccade blocks'),(ax[1],'anti','Antisaccade blocks')]:
    a_.set_ylim(60,104); ys=[];ns=[]
    for p in [1,4,6]:
        d=b[(b.participant==p)&(b.task==task)].set_index('stage')
        c=d.condition.iloc[0]; v=[d.loc['pre','peripheral_accuracy_among_responses']*100,d.loc['post','peripheral_accuracy_among_responses']*100]
        a_.plot([0,1],v,color=col(c),marker=mk(c),ms=7,mec='white')
        ys.append(v[1]); ns.append(f"P00{p} ({int(d.loc['pre','peripheral_responses'])}→{int(d.loc['post','peripheral_responses'])} resp.)")
    lab(a_,1,ys,ns)
    a_.set_xticks([0,1]); a_.set_xticklabels(['Pre','Post']); a_.set_xlim(-0.2,1.9); a_.set_title(tt,fontsize=11)
ax[0].set_ylabel('Peripheral bracket accuracy\n(% of recorded responses)')
from matplotlib.lines import Line2D
ax[0].legend([Line2D([],[],color=INC,marker='o'),Line2D([],[],color=EXC,marker='s')],['Inclusion','Exclusion'],frameon=False,loc='lower left',fontsize=9)
f.tight_layout(); f.savefig('figures/fig4_peripheral_accuracy.png',dpi=200); plt.close()
# Fig 5 questionnaires paired n=6
qq=q.dropna(subset=['shame_pre_mean','shame_post_mean'])
f,ax=plt.subplots(1,2,figsize=(8,3.6),sharey=True)
for a_,m,tt in [(ax[0],'shame','State shame'),(ax[1],'belonging','Belongingness')]:
    a_.set_ylim(1,5); ys=[];ns=[]
    for _,r in qq.iterrows():
        v=[r[m+'_pre_mean'],r[m+'_post_mean']]
        a_.plot([0,1],v,color=col(r.condition),marker=mk(r.condition),ms=7,mec='white',alpha=.9)
        ys.append(v[1]); ns.append(f"P00{int(r.participant)}")
    lab(a_,1,ys,ns,gap=0.17)
    a_.set_xticks([0,1]); a_.set_xticklabels(['Pre','Post']); a_.set_xlim(-0.2,1.45); a_.set_title(tt,fontsize=11)
ax[0].set_ylabel('Item mean (1–5)')
ax[1].legend([Line2D([],[],color=INC,marker='o'),Line2D([],[],color=EXC,marker='s')],['Inclusion (n = 2)','Exclusion (n = 4)'],frameon=False,loc='upper left',fontsize=9)
f.tight_layout(); f.savefig('figures/fig5_shame_belonging.png',dpi=200); plt.close()
# Fig 6 eye candidate sensitivity
f,ax=plt.subplots(1,2,figsize=(8,3.3),sharey=True)
for a_,met,tt in [(ax[0],'candidate_rate','Trials with a candidate shift\nduring the 300-ms cue'),(ax[1],'cue_direction_match_among_candidates','Candidates directed toward the cue\n(% of candidates)')]:
    for task,c,m,nm in [('pro',INC,'o','Prosaccade (215 trials)'),('anti',EXC,'s','Antisaccade (190 trials)')]:
        d=s[s.task==task]; a_.plot(d.threshold,d[met]*100,color=c,marker=m,ms=7,mec='white',label=nm)
    a_.set_xticks([30,40,60]); a_.set_xlabel('Velocity threshold (°/s)'); a_.set_title(tt,fontsize=10); a_.set_ylim(0,105)
ax[0].set_ylabel('%'); ax[0].legend(frameon=False,fontsize=8.5,loc='lower left')
f.tight_layout(); f.savefig('figures/fig6_eye_candidate_sensitivity.png',dpi=200); plt.close()
