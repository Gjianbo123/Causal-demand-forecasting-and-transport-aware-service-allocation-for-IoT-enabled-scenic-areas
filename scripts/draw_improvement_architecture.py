"""Editable vector schematic of the implemented v4 system, without field claims."""
from pathlib import Path
import os
ROOT=Path(__file__).resolve().parents[1];os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'tmp/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
import sys
sys.path.insert(0,str(ROOT))
from iotexp.scenic import scenic_graph

def main():
    out=ROOT/'output/improvement_v4/figures';out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'Arial','font.size':8.5,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','svg.hashsalt':'iot-v4'})
    fig,ax=plt.subplots(figsize=(165/25.4,165/25.4));fig.subplots_adjust(0,0,1,1)
    ax.set(xlim=(0,165),ylim=(0,165));ax.axis('off');blue='#2C6282';green='#216C64';ink='#253448'
    def text(x,y,s,size=8.5,**kw):ax.text(x,y,s,ha='center',va='center',fontsize=size,color=ink,linespacing=1.3,**kw)
    def box(x,y,w,h,title,body,color=blue,fill='white'):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=2',facecolor=fill,edgecolor=color,lw=.8))
        text(x+w/2,y+h-5,title,9,fontweight='bold');text(x+w/2,y+h/2-3,body)
    def arrow(a,b,label=None):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=9,color=ink,lw=.9))
        if label:text((a[0]+b[0])/2,(a[1]+b[1])/2+3,label,8)
    ax.add_patch(FancyBboxPatch((5,126),155,36,boxstyle='round,pad=0,rounding_size=2',facecolor='#F3F6FA',edgecolor='#C6D1DA'))
    text(82.5,157,'OFFLINE DEVELOPMENT AND FREEZE',9,fontweight='bold')
    box(9,130,45,21,'Training data','Slot templates\nResidual GRU + GWN')
    box(60,130,45,21,'Validation','Select forecast blend\nSelect rollout horizon')
    box(111,130,45,21,'Sealed evaluation','Freeze code and weights\nSeven new random worlds')
    arrow((54,140),(60,140));arrow((105,140),(111,140))
    text(82.5,117,'CAUSAL DECISION LOOP: EVERY 10 MINUTES',9,fontweight='bold')
    box(8,70,44,39,'Telemetry','12 observed inflow frames\nCurrent service queues\nEvent / receipt times\nTrusted resource ledger',fill='#F0F7FA')
    box(60,70,45,39,'Forecasting','Training-slot anchor\nGraph residual GRU\nGWN forecast combination\nLeads: 10/20/40/60 min',fill='#F0F7FA')
    box(113,70,44,39,'Transport rollout','Queue-cost reduction\nIn-transit service loss\n12-step look-ahead\nOne-unit marginal moves',green,'#F0F8F4')
    arrow((52,89),(60,89));arrow((105,89),(113,89))
    arrow((30,109),(30,112));arrow((30,112),(135,112));arrow((135,112),(135,109))
    text(82.5,114,'Current queues and trusted ledger',8)
    box(8,16,44,40,'Exogenous process','Synthetic visit events\nFixed request realization\nShared across controllers\nNo action-to-inflow link',fill='#F9F7F0')
    box(60,16,45,40,'Service simulator','Integer resource inventory\nCommitted transport\nFIFO service queues\nClosing-time backlog',green,'#F0F8F4')
    box(113,16,44,40,'Executable dispatch','Origin-destination flows\nConservation / holding\nNo in-transit service\nNo rerouting commitments',green,'#F0F8F4')
    arrow((135,70),(135,56));arrow((113,36),(105,36));arrow((52,36),(60,36))
    arrow((82.5,56),(82.5,63));arrow((82.5,63),(30,63));arrow((30,63),(30,70))
    text(68,66,'Observed counts and resource feedback',8)
    text(82.5,7,'Fixed physical graph: forecasting support and shortest-path transport',8.3)
    for suffix in ['pdf','svg','png','eps']:
        fig.savefig(out/f'System_architecture.{suffix}',dpi=240)
    plt.close(fig)
    # Exact simulator topology; an outer return avoids crossing service nodes.
    data=scenic_graph();fig,ax=plt.subplots(figsize=(165/25.4,85/25.4))
    fig.subplots_adjust(.025,.04,.975,.97);ax.set(xlim=(-1.8,13.8),ylim=(-2.5,9));ax.axis('off');positions={}
    for cluster in range(4):
        x=cluster*4
        for local,(dx,y) in enumerate([(-1,8),(1,8),(-1,5.8),(1,5.8)]):positions[4*cluster+local]=(x+dx,y)
        positions[16+cluster]=(x,2.4);positions[20+cluster]=(x,.2)
    for i,j in data['edge_index']:
        a,b=positions[i],positions[j]
        if {int(i),int(j)}=={16,19}:
            ax.plot([0,-1.5,-1.5,13.5,13.5,12],[2.4,2.4,-1.5,-1.5,2.4,2.4],color='#75848B',lw=1,zorder=1)
        else:ax.plot([a[0],b[0]],[a[1],b[1]],color='#75848B',lw=1,zorder=1)
    for i,(x,y) in positions.items():
        marker,color,label=('o','#D9EBCF',f'S{i+1}') if i<16 else (('s','#DCE8F3',f'H{i-15}') if i<20 else ('D','#F4E4BA',f'G{i-19}'))
        ax.scatter(x,y,s=400 if i<16 else 330,marker=marker,facecolor=color,edgecolor='#455C6A',lw=.8,zorder=2)
        ax.text(x,y,label,ha='center',va='center',fontsize=8.5,zorder=3)
    ax.text(6,-2.15,'S: service location    H: hub    G: gate    Links are undirected',ha='center',va='center',fontsize=8.5)
    for suffix in ['pdf','svg','png','eps']:fig.savefig(out/f'Scenic_graph.{suffix}',dpi=240)
    plt.close(fig)

if __name__=='__main__':main()
