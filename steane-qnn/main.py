"""Steane decoding, classical and quantum training, and gradient experiments."""


# STEANE
"""Steane CSS code; qubit zero is the most significant statevector bit."""
import itertools
import numpy as np

H = np.array([[0,0,0,1,1,1,1],[0,1,1,0,0,1,1],[1,0,1,0,1,0,1]],dtype=int)
SPAN = np.array(list(itertools.product([0,1],repeat=3))) @ H % 2
W = np.array([4,2,1])
POW7 = 2**np.arange(6,-1,-1)

def syndrome(x,z):
    """Hx (Z checks), then Hz (X checks). Pauli phases are irrelevant."""
    return np.concatenate([np.asarray(x)@H.T%2,np.asarray(z)@H.T%2],axis=-1)

def lookup(s):
    s=np.asarray(s,dtype=int)
    return np.stack([s[...,:3]@W,s[...,3:]@W],axis=-1)

def corrections(labels):
    labels=np.asarray(labels).reshape(-1,2)
    x=np.zeros((len(labels),7),dtype=int); z=np.zeros_like(x)
    for k,out in enumerate([x,z]):
        active=labels[:,k]>0
        out[np.where(active)[0],labels[active,k]-1]=1
    return x,z

def stabilizer_member(x,z):
    def inside(a): return (a[:,None,:]==SPAN[None,:,:]).all(-1).any(-1)
    return inside(np.asarray(x)) & inside(np.asarray(z))

def wilson(k,n):
    a=1.95996398454; p=k/n; den=1+a*a/n
    mid=(p+a*a/(2*n))/den
    half=a*np.sqrt(p*(1-p)/n+a*a/(4*n*n))/den
    return [float(mid-half),float(mid+half)]

def metrics(x,z,labels):
    cx,cz=corrections(labels); rx,rz=x^cx,z^cz
    failed=~stabilizer_member(rx,rz)
    detectable=syndrome(rx,rz).any(-1)
    fx,fz=corrections(lookup(syndrome(rx,rz)))
    logical=~stabilizer_member(rx^fx,rz^fz)
    return dict(recovery_failure=float(failed.mean()),detectable_residual=float(detectable.mean()),
        logical_error_after_ideal_cleanup=float(logical.mean()),logical_errors=int(logical.sum()),
        count=len(x),logical_ci95=wilson(int(logical.sum()),len(x)))

def encode(alpha=1/np.sqrt(2),beta=1j/np.sqrt(2)):
    if not np.isclose(abs(alpha)**2+abs(beta)**2,1): raise ValueError('Normalize amplitudes')
    state=np.zeros(128,dtype=complex); indices=SPAN@POW7
    state[indices]=alpha/np.sqrt(8); state[indices^127]=beta/np.sqrt(8)
    return state

def apply_pauli(state,x,z):
    indices=np.arange(128); bits=(indices[:,None]>>np.arange(6,-1,-1))&1
    out=np.empty_like(state)
    out[indices^int(np.asarray(x)@POW7)]=state*(-1.)**(bits@np.asarray(z)%2)
    return out

def measure_stabilizers(state,seed=0):
    rng=np.random.default_rng(seed); state=state.copy(); result=[]; zero=np.zeros(7,dtype=int)
    for x,z in [(zero,h) for h in H]+[(h,zero) for h in H]:
        transformed=apply_pauli(state,x,z)
        pplus=np.clip((1+np.real(np.vdot(state,transformed)))/2,0,1)
        bit=int(rng.random()>=pplus)
        state=state+(-1)**bit*transformed; state/=np.linalg.norm(state)
        result.append(bit)
    return np.array(result),state

def dataset(count,rounds,p,q,seed):
    """Independent phenomenological trajectories, no recovery between rounds."""
    rng=np.random.default_rng(seed)
    if p is None: p=rng.choice([.02,.06,.12],size=(count,1))
    if q is None: q=rng.choice([0.,.03,.08],size=(count,1))
    x=np.zeros((count,7),dtype=int); z=np.zeros_like(x)
    observations=[]; truths=[]; dxs=[]; dzs=[]
    for _ in range(rounds):
        event=rng.random((count,7))<p; kind=rng.integers(1,4,size=(count,7))
        dx=(event&((kind==1)|(kind==2))).astype(int)
        dz=(event&((kind==2)|(kind==3))).astype(int)
        x^=dx; z^=dz; true=syndrome(x,z)
        observed=true^(rng.random((count,6))<q).astype(int)
        observations.append(observed); truths.append(true); dxs.append(dx); dzs.append(dz)
    return dict(observed=np.stack(observations,1),true_syndromes=np.stack(truths,1),
        x=x,z=z,labels=lookup(syndrome(x,z)),increments_x=np.stack(dxs,1),
        increments_z=np.stack(dzs,1),p=np.broadcast_to(p,(count,1)).copy(),
        q=np.broadcast_to(q,(count,1)).copy())


# MODELS
"""Exact statevector QNN with genuine RZZ entanglers and local readout."""
import numpy as np
import torch
from torch import nn
D=torch.float64

def rotation(t,axis):
    c,s=torch.cos(t/2),torch.sin(t/2)
    if axis=='Y': return torch.stack([torch.stack([c,-s]),torch.stack([s,c])]).to(torch.complex128)
    if axis=='X': return torch.stack([torch.stack([c+0j,-1j*s]),torch.stack([-1j*s,c+0j])])
    zero=0j*c
    return torch.stack([torch.stack([torch.exp(-.5j*t),zero]),torch.stack([zero,torch.exp(.5j*t)])])

def gate(state,matrix,q,n):
    shaped=state.reshape([len(state)]+[2]*n).movedim(q+1,-1)
    return (shaped@matrix.T).movedim(-1,q+1).reshape(len(state),2**n)

def signs(n):
    bits=(torch.arange(2**n)[:,None]>>torch.arange(n-1,-1,-1))&1
    return (1-2*bits).to(D)

def prepare(bits):
    n=bits.shape[-1]; index=(bits.long()*2**torch.arange(n-1,-1,-1)).sum(-1)
    state=torch.zeros((len(bits),2**n),dtype=torch.complex128)
    state[torch.arange(len(bits)),index]=1
    for q in range(n):
        state=gate(state,rotation(torch.tensor(.37,dtype=D),'Y'),q,n)
        state=gate(state,rotation(torch.tensor(.41,dtype=D),'X'),q,n)
    return state

def initialize(n,depth,kind,seed):
    generator=torch.Generator().manual_seed(seed); shape=(depth,3*n-1)
    if kind=='small': return .05*torch.randn(shape,generator=generator,dtype=D)
    return (torch.rand(shape,generator=generator,dtype=D)*2-1)*np.pi

def evolve(theta,state,n):
    z=signs(n)
    for layer in theta:
        for q in range(n):
            state=gate(state,rotation(layer[2*q],'Y'),q,n)
            state=gate(state,rotation(layer[2*q+1],'Z'),q,n)
        for q in range(n-1):
            state=state*torch.exp(-.5j*layer[2*n+q]*z[:,q]*z[:,q+1])
    # Fixed basis change exposes final diagonal entanglers in measured features.
    for q in range(n): state=gate(state,rotation(torch.tensor(.43,dtype=D),'Y'),q,n)
    return state

def features(state,n):
    z=signs(n); observables=torch.cat([z,z[:,:-1]*z[:,1:]],dim=1)
    return state.abs().square()@observables

class Decoder(nn.Module):
    def __init__(self,kind,depth=3,seed=11):
        super().__init__(); self.kind=kind
        if kind=='mlp': self.network=nn.Sequential(nn.Linear(6,32),nn.Tanh(),nn.Linear(32,16)).double()
        elif kind=='gru':
            self.recurrent=nn.GRU(6,24,batch_first=True).double(); self.head=nn.Linear(24,16).double()
        else:
            init='small' if 'small' in kind else 'random'
            self.theta=nn.Parameter(initialize(6,depth,init,seed),requires_grad=('frozen' not in kind))
            bits=torch.tensor([[int(c) for c in f'{i:06b}'] for i in range(64)])
            self.register_buffer('states',prepare(bits)); self.head=nn.Linear(11,16).double()
    def forward(self,sequence):
        if self.kind=='mlp': result=self.network(2*sequence[:,-1]-1)
        elif self.kind=='gru':
            output,_=self.recurrent(2*sequence-1); result=self.head(output[:,-1])
        else:
            index=(sequence[:,-1].long()*2**torch.arange(5,-1,-1)).sum(-1)
            unique,inverse=torch.unique(index,return_inverse=True)
            state=evolve(self.theta,self.states[unique],6)
            result=self.head(features(state,6))[inverse]
        return result.reshape(-1,2,8)


# TRAIN
"""Train models and evaluate common independent physical-error trajectories."""
import argparse,csv,json,time,platform
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

def csv_write(path,rows):
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def latency(model,x,repeats=50):
    measurements=[]
    with torch.no_grad():
        for _ in range(5): model(x[:1])
        for _ in range(repeats):
            start=time.perf_counter(); model(x[:1]); measurements.append((time.perf_counter()-start)*1000)
    return float(np.median(measurements))

def train_main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',default='results/main'); p.add_argument('--epochs',type=int,default=120)
    p.add_argument('--models',nargs='+',default=['mlp','gru','qnn_random','qnn_small','qnn_small_frozen'])
    p.add_argument('--seeds',type=int,nargs='+',default=[11,22,33])
    p.add_argument('--depth',type=int,default=3); p.add_argument('--lr',type=float,default=.03)
    p.add_argument('--train-size',type=int,default=4096); p.add_argument('--test-size',type=int,default=4096)
    p.add_argument('--task',choices=['temporal','table'],default='temporal')
    a=p.parse_args(); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    (out/'checkpoints').mkdir(exist_ok=True)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    train=dataset(a.train_size,3,None,None,1001); valid=dataset(1024,3,None,None,1002)
    if a.task=='table':
        s=np.array([[int(c) for c in f'{i:06b}'] for i in range(64)])
        train=dict(observed=s[:,None,:],labels=lookup(s)); valid=train
    np.savez_compressed(out/'train.npz',**train); np.savez_compressed(out/'validation.npz',**valid)
    xt=torch.tensor(train['observed'],dtype=torch.float64); xv=torch.tensor(valid['observed'],dtype=torch.float64)
    yt=torch.tensor(train['labels']); yv=torch.tensor(valid['labels'])
    def loss(logits,y): return F.cross_entropy(logits.reshape(-1,8),y.reshape(-1))
    tests={(p,q):dataset(a.test_size,3,p,q,2000+i*10+j)
           for i,p in enumerate([.01,.05,.10]) for j,q in enumerate([0.,.03,.08])}
    # All models see identical evaluation trajectories, including across training seeds.
    evaluations=[]; summaries=[]; history=[]
    for (p,q),data in tests.items():
        evaluations.append(dict(model='lookup',seed=0,p=p,q=q,**metrics(data['x'],data['z'],lookup(data['observed'][:,-1]))))
    timings=[]
    for _ in range(100):
        start=time.perf_counter(); lookup(valid['observed'][:1,-1]); timings.append((time.perf_counter()-start)*1000)
    summaries.append(dict(model='lookup',seed=0,training_seconds=0,parameters=0,quantum_parameters=0,
        latency_ms=float(np.median(timings)),final_train_loss=None,final_validation_loss=None,
        validation_accuracy=None,epoch_to_loss_08=None))
    for seed in a.seeds:
        for kind in a.models:
            torch.manual_seed(seed); model=Decoder(kind,a.depth,seed)
            optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=a.lr)
            start=time.perf_counter(); threshold=None
            for epoch in range(a.epochs+1):
                qgrad=0.
                if epoch:
                    optimizer.zero_grad(); value=loss(model(xt),yt); value.backward()
                    if hasattr(model,'theta') and model.theta.grad is not None: qgrad=float(model.theta.grad.norm())
                    optimizer.step()
                with torch.no_grad():
                    tl=loss(model(xt),yt).item(); logits=model(xv); vl=loss(logits,yv).item()
                    acc=(logits.argmax(-1)==yv).all(-1).double().mean().item()
                if vl<=.8 and threshold is None: threshold=epoch
                history.append(dict(model=kind,seed=seed,epoch=epoch,train_loss=tl,validation_loss=vl,
                    validation_accuracy=acc,quantum_gradient_norm=qgrad,seconds=time.perf_counter()-start))
            elapsed=time.perf_counter()-start
            row=dict(model=kind,seed=seed,training_seconds=elapsed,
                parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                quantum_parameters=(model.theta.numel() if hasattr(model,'theta') and model.theta.requires_grad else 0),
                latency_ms=latency(model,xv),final_train_loss=tl,final_validation_loss=vl,
                validation_accuracy=acc,epoch_to_loss_08=threshold)
            summaries.append(row)
            torch.save(dict(kind=kind,depth=a.depth,seed=seed,state_dict=model.state_dict()),out/'checkpoints'/f'{kind}_{seed}.pt')
            with torch.no_grad():
                for (p,q),data in tests.items():
                    predictions=model(torch.tensor(data['observed'],dtype=torch.float64)).argmax(-1).numpy()
                    evaluations.append(dict(model=kind,seed=seed,p=p,q=q,**metrics(data['x'],data['z'],predictions)))
            print(json.dumps(row),flush=True)
            csv_write(out/'history.csv',history); csv_write(out/'models.csv',summaries)
            (out/'evaluation.json').write_text(json.dumps(evaluations,indent=2))
            (out/'summary.json').write_text(json.dumps(summaries,indent=2))
    (out/'config.json').write_text(json.dumps(vars(a),indent=2))
    (out/'environment.json').write_text(json.dumps(dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,platform=platform.platform()),indent=2))



# PLATEAU
"""Fixed-component gradient variance with independent repeated initializations."""
import argparse,json
from pathlib import Path
import numpy as np
import torch

def plateau_main():
    p=argparse.ArgumentParser(); p.add_argument('--out',default='results/plateau')
    p.add_argument('--qubits',type=int,nargs='+',default=list(range(2,11)))
    p.add_argument('--depths',type=int,nargs='+',default=[1,3,6]); p.add_argument('--samples',type=int,default=128)
    p.add_argument('--seed',type=int,default=71); a=p.parse_args()
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True); torch.set_num_threads(1)
    raw=[]; stats=[]; rng=np.random.default_rng(a.seed)
    for depth in a.depths:
        for n in a.qubits:
            initial=prepare(torch.zeros((1,n)))
            groups={(k,c):[] for k in ['random','small'] for c in ['global','local']}
            for sample in range(a.samples):
                seed=a.seed+100000*depth+1000*n+sample
                for kind in ['random','small']:
                    theta=initialize(n,depth,kind,seed).requires_grad_(); state=evolve(theta,initial,n)
                    costs=dict(global_cost=1-state[:,0].abs().square().mean(),local=(1-features(state,n)[:,:n]).mean()/2)
                    for cost_name,cost in costs.items():
                        name='global' if cost_name=='global_cost' else cost_name
                        grad=torch.autograd.grad(cost,theta,retain_graph=True)[0][0,0].item()
                        groups[kind,name].append(grad)
                        raw.append(dict(n=n,depth=depth,initialization=kind,cost=name,sample=sample,gradient=grad))
            for (kind,cost),values in groups.items():
                values=np.array(values); boot=np.var(rng.choice(values,(500,len(values)),replace=True),axis=1,ddof=1)
                lo,hi=np.quantile(boot,[.025,.975])
                stats.append(dict(n=n,depth=depth,initialization=kind,cost=cost,variance=float(values.var(ddof=1)),ci_low=float(lo),ci_high=float(hi),mean_gradient=float(values.mean())))
            csv_write(out/'samples.csv',raw); csv_write(out/'variance.csv',stats)
            print(f'Gradient variance n={n}, depth={depth} finished',flush=True)
    slopes=[]
    for depth in a.depths:
        for kind in ['random','small']:
            for cost in ['global','local']:
                rows=[r for r in stats if r['depth']==depth and r['initialization']==kind and r['cost']==cost]
                slopes.append(dict(depth=depth,initialization=kind,cost=cost,
                    log10_slope=float(np.polyfit([r['n'] for r in rows],np.log10([max(r['variance'],1e-300) for r in rows]),1)[0])))
    (out/'slopes.json').write_text(json.dumps(slopes,indent=2)); (out/'config.json').write_text(json.dumps(vars(a),indent=2))



# FIGURES
"""Generate publication-style figures from measured CSV/JSON files."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS={'lookup':'#52525b','mlp':'#0369a1','gru':'#0f766e','qnn_random':'#dc2626','qnn_small':'#7c3aed','qnn_small_frozen':'#d97706'}
LABELS={'lookup':'Lookup','mlp':'MLP','gru':'Temporal GRU','qnn_random':'QNN random','qnn_small':'QNN small-angle','qnn_small_frozen':'QNN frozen control'}

def rows(path):
    with open(path) as f: return list(csv.DictReader(f))

def figures_main():
    out=Path('results/figures'); out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
    for task in ['main','table']:
        hist=rows(f'results/{task}/history.csv'); config=json.loads(Path(f'results/{task}/config.json').read_text())
        fig,axes=plt.subplots(1,2,figsize=(10.8,3.8))
        for model in config['models']:
            runs=[[r for r in hist if r['model']==model and int(r['seed'])==seed] for seed in config['seeds']]
            values=np.array([[float(r['validation_loss']) for r in run] for run in runs])
            mean,std=values.mean(0),values.std(0); x=np.arange(len(mean)); color=COLORS[model]
            axes[0].plot(x,mean,label=LABELS[model],color=color)
            axes[0].fill_between(x,np.maximum(mean-std,1e-10),mean+std,color=color,alpha=.12)
            for i,run in enumerate(runs):
                axes[1].plot([float(r['seconds']) for r in run],[float(r['validation_loss']) for r in run],color=color,alpha=.5,label=LABELS[model] if i==0 else None)
        for ax in axes:
            ax.set_yscale('log'); ax.grid(alpha=.2); ax.set_ylabel('Cross-entropy'); ax.legend(fontsize=9)
        axes[0].set_xlabel('Optimizer updates'); axes[1].set_xlabel('CPU training seconds')
        axes[0].set_title('Mean and one SD across seeds'); axes[1].set_title('Individual runs')
        fig.tight_layout(); fig.savefig(out/f'{task}_convergence.png',dpi=200); plt.close(fig)
    data=json.loads(Path('results/main/evaluation.json').read_text())
    fig,axes=plt.subplots(1,3,figsize=(11,3.6),sharey=True)
    for ax,q in zip(axes,[0.,.03,.08]):
        for model in COLORS:
            mean=[]; std=[]
            for p in [.01,.05,.10]:
                values=[r['logical_error_after_ideal_cleanup'] for r in data if r['model']==model and r['p']==p and r['q']==q]
                mean.append(np.mean(values)); std.append(np.std(values))
            mean=np.array(mean); std=np.array(std)
            ax.plot([.01,.05,.1],mean,'o-',color=COLORS[model],label=LABELS[model],markersize=4)
            ax.fill_between([.01,.05,.1],np.maximum(0,mean-std),mean+std,alpha=.1,color=COLORS[model])
        ax.set_title(f'Readout flip q = {q}'); ax.set_xlabel('Data error probability per round'); ax.grid(alpha=.2)
    axes[0].set_ylabel('Logical error after ideal cleanup'); axes[-1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out/'logical_error.png',dpi=200); plt.close(fig)
    stats=rows('results/plateau/variance.csv'); depths=sorted(set(int(r['depth']) for r in stats))
    fig,axes=plt.subplots(1,len(depths),figsize=(11,3.6),squeeze=False)
    for ax,depth in zip(axes[0],depths):
        for kind,color in [('random','#dc2626'),('small','#7c3aed')]:
            for cost,style in [('global','-'),('local','--')]:
                rs=sorted([r for r in stats if int(r['depth'])==depth and r['initialization']==kind and r['cost']==cost],key=lambda r:int(r['n']))
                x=[int(r['n']) for r in rs]; y=[float(r['variance']) for r in rs]
                ax.plot(x,y,style,marker='o',markersize=3,color=color,label=f'{kind}/{cost}')
                ax.fill_between(x,[float(r['ci_low']) for r in rs],[float(r['ci_high']) for r in rs],color=color,alpha=.10)
        ax.set_yscale('log'); ax.set_xlabel('Qubits'); ax.set_title(f'Depth {depth}'); ax.grid(alpha=.2); ax.legend(fontsize=8.5)
    axes[0,0].set_ylabel('Fixed-component gradient variance')
    fig.tight_layout(); fig.savefig(out/'gradient_variance.png',dpi=200); plt.close(fig)
    n=np.arange(2,31); d=np.arange(1,101)
    fig,axes=plt.subplots(1,2,figsize=(10,3.4))
    axes[0].semilogy(n,.125*.375**(n-1),label='Global: exponential',color='#dc2626')
    axes[0].semilogy(n,1/(8*n*n),label='Local: polynomial',color='#7c3aed')
    axes[0].set(xlabel='Qubits',ylabel='Exact gradient variance',title='Separate depth-one RY product circuit')
    for noise in [.01,.05,.1]: axes[1].semilogy(d,(1-noise)**(2*d),label=f'p={noise}')
    axes[1].set(xlabel='Noisy layers',ylabel='Noisy / clean variance',title='Analytical GLOBAL depolarizing model')
    for ax in axes: ax.legend(fontsize=8); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(out/'mechanism_noise.png',dpi=200); plt.close(fig)



# VERIFY SAVED NUMERICAL RESULTS
"""Check saved results without retraining. Run from the repository root."""
import csv,json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

def verify_main():
    torch.set_num_threads(1)
    verified=0
    for task in ['main','table']:
        root=Path('results')/task
        config=json.loads((root/'config.json').read_text())
        data=np.load(root/'validation.npz')
        x=torch.tensor(data['observed'],dtype=torch.float64); y=torch.tensor(data['labels'])
        rows=list(csv.DictReader((root/'history.csv').open()))
        assert len(rows)==len(config['models'])*len(config['seeds'])*(config['epochs']+1)
        for result in json.loads((root/'summary.json').read_text()):
            if result['model']=='lookup': continue
            saved=torch.load(root/'checkpoints'/f'{result["model"]}_{result["seed"]}.pt',map_location='cpu',weights_only=True)
            model=Decoder(saved['kind'],saved['depth'],saved['seed'])
            model.load_state_dict(saved['state_dict']); model.eval()
            with torch.no_grad():
                logits=model(x); loss=F.cross_entropy(logits.reshape(-1,8),y.reshape(-1)).item()
                accuracy=(logits.argmax(-1)==y).all(-1).double().mean().item()
            assert abs(loss-result['final_validation_loss'])<1e-10
            assert abs(accuracy-result['validation_accuracy'])<1e-12
            verified+=1
    raw=list(csv.DictReader(open('results/plateau/samples.csv')))
    stats=list(csv.DictReader(open('results/plateau/variance.csv')))
    cfg=json.loads(Path('results/plateau/config.json').read_text())
    assert len(raw)==len(cfg['qubits'])*len(cfg['depths'])*cfg['samples']*4
    for row in stats:
        values=[float(r['gradient']) for r in raw if all(r[k]==row[k] for k in ['n','depth','initialization','cost'])]
        assert len(values)==cfg['samples']
        assert np.isclose(np.var(values,ddof=1),float(row['variance']),rtol=1e-12,atol=0)
    print(f"Verified {verified} checkpoints, {len(raw)} gradient samples, and {len(stats)} variance groups.")


def cli():
    import sys
    import subprocess
    commands = {"train": train_main, "plateau": plateau_main,
                "plot": figures_main, "verify": verify_main}
    if len(sys.argv) < 2 or sys.argv[1] in ["-h", "--help"]:
        print("Usage: python main.py {all|train|plateau|plot|verify} [options]")
        print("Use train --help or plateau --help for experiment options.")
        return
    command = sys.argv.pop(1)
    if command == "all":
        runs = [["train"], ["train", "--task", "table", "--models", "mlp",
                "qnn_random", "qnn_small", "qnn_small_frozen", "--out", "results/table"],
                ["plateau"], ["plot"], ["verify"]]
        for args in runs:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), *args], check=True)
    elif command in commands:
        commands[command]()
    else:
        raise SystemExit(f"Unknown command: {command}")

if __name__ == "__main__":
    cli()
