"""Sanity check of the Liang re-implementation: plant = the same LPTN (exact model), real inputs of V1/V2,
true initial state = real measured temperatures at the reset; measurement = simulated winding (+ optional noise)."""
import sys, json, importlib.util
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location("lc",ROOT/"scripts/run_liang_comparison.py")
src=(ROOT/"scripts/run_liang_comparison.py").read_text(encoding="utf-8").split("rows = []")[0]   # definitions only
ns={"__file__":str(ROOT/"scripts/run_liang_comparison.py")}; exec(compile(src,"lc","exec"),ns)
sim,cache,VAL,liang_estimate,TS=ns["sim"],ns["cache"],ns["VAL"],ns["liang_estimate"],ns["TS"]
rng=np.random.default_rng(1); out={}
for noise in [0.0,0.1,0.3]:
    errs={1:[],4:[]}
    for pid in VAL:
        o,F,Y=cache[pid]
        for r in range(0,len(o)-1200,1800):
            Ysim=Y.copy(); n=min(len(o),r+1000)
            tr=sim.rollout(Y[r],r,n,F); Ysim[r:n]=tr; Ysim[r+1:n,3]+=rng.normal(0,noise,n-r-1) if noise>0 else 0
            for w in errs:
                xh=liang_estimate(Ysim,F,r,int(w*60/TS)); errs[w].append(abs(xh[0]-Y[r,0]))
    out[f"noise={noise}"]={f"W={w}min_pm0_med":float(np.median(e)) for w,e in errs.items()}
print(json.dumps(out,indent=1)); (ROOT/"outputs/analysis").mkdir(parents=True,exist_ok=True); (ROOT/"outputs/analysis"/"liang_sanity_synthetic.json").write_text(json.dumps(out,indent=1))
