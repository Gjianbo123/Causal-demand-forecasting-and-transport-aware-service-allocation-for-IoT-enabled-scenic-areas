"""Independent read-only numeric audit of the supplementary PPO outputs."""
import argparse,json,itertools,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from iotexp.supplement_common import V2,V3,verify_v2,scenarios,make_scenario_backend
from iotexp.util import digest,write_json


def close(x,y,label):
    if not np.allclose(x,y,atol=1e-5,rtol=1e-7,equal_nan=True):raise AssertionError(label)


def audit(partial=False):
    start=time.perf_counter();verify_v2()
    protocol=json.loads((V3/"protocol.json").read_text())
    for rel,value in protocol["source_files"].items():assert digest(ROOT/rel)==value
    datasets={}
    for name,seed,surge in scenarios():
        b=make_scenario_backend(seed,surge)
        datasets[name]={"arrivals":b.data["demand"][153:,15:].sum((1,2,3)),"seed":seed,"surge":surge}
    metadata={};trained=0;budget_prefix_comparisons=0
    for predictor,methods in [("edge_stgru",["F0","F1","F4"]),("ridge",["F1","F4"])]:
        for method,seed in itertools.product(methods,range(10)):
            folder=V3/"policies"/predictor/"checkpoints"/f"{method}_seed{seed}"
            if not (folder/"metadata.json").exists():
                assert partial;continue
            info=json.loads((folder/"metadata.json").read_text())
            logs=[json.loads(line) for line in (folder/"training.jsonl").read_text().splitlines()]
            validation=[v for v in logs if "validation_selection_score" in v]
            assert len(validation)==12
            winner=min(validation,key=lambda v:v["validation_selection_score"])
            assert info["total_steps"]==43200 and logs[-1]["completed_episodes"]==600
            assert info["method"]==method and info["training_seed"]==seed
            assert info["selection_rule"]=="validation_reward" and info["best_step"]==winner["step"]
            close(info["validation_selection_score"],winner["validation_selection_score"],"selected validation checkpoint")
            assert info["checkpoint_sha256"]==digest(folder/"best.pt")
            metadata[predictor,method,seed,43200]=info;trained+=1
            if predictor=="edge_stgru":
                prior=V2/"checkpoints"/f"{method}_seed{seed}"
                old_logs=[json.loads(line) for line in (prior/"training.jsonl").read_text().splitlines()]
                old_scores={v["step"]:v["validation_selection_score"] for v in old_logs if "validation_selection_score" in v}
                for item in validation:
                    if item["step"] in old_scores:
                        close(item["validation_selection_score"],old_scores[item["step"]],"same seeded first14400 training prefix")
                        budget_prefix_comparisons+=1
    for method,seed in itertools.product(["F0","F1","F4"],range(10)):
        folder=V2/"checkpoints"/f"{method}_seed{seed}"
        metadata["edge_stgru",method,seed,14400]=json.loads((folder/"metadata.json").read_text())
    paths=list((V3/"policies").glob("*/evaluation_shards/*.csv"))
    paths=[p for p in paths if ".partial." not in p.name]
    assert paths
    if not partial:assert trained==50 and len(paths)==80
    frames=[]
    for path in paths:
        df=pd.read_csv(path)
        assert len(df)==189 and not df.duplicated(["scenario","environment_seed"]).any()
        assert set(zip(df.scenario,df.environment_seed))=={(s[0],day) for s in scenarios() for day in range(27)}
        frames.append(df)
    episodes=pd.concat(frames,ignore_index=True)
    keys=["predictor","method","training_seed","budget_steps","scenario","environment_seed"]
    assert not episodes.duplicated(keys).any()
    raw_samples=0;trace_count=0
    for row in episodes.itertuples():
        source=datasets[row.scenario]
        assert row.dataset_seed==source["seed"] and row.surge==source["surge"]
        assert row.arrivals_count==source["arrivals"][row.environment_seed]
        meta=metadata[row.predictor,row.method,row.training_seed,row.budget_steps]
        assert row.checkpoint_sha256==meta["checkpoint_sha256"] and row.best_step==meta["best_step"]
        if hasattr(row,"trainable_parameters"):
            assert row.trainable_parameters==meta["trainable_parameters"]
        assert row.decision_epochs==72 and row.executable_infeasible_epochs==0
        assert row.served_count+row.unserved_count==row.arrivals_count
        with np.load(ROOT/row.trace_path,allow_pickle=False) as z:
            assert np.array_equal(z["decision_time"],np.arange(72))
            waits=z["served_waits_min"]
            assert len(waits)==row.served_count and (waits>=0).all() and (waits%10==0).all()
            close(waits.sum(),row.wait_sum_min,"raw wait sum")
            close(waits.mean(),row.mean_wait_min,"raw wait mean")
            close(np.percentile(waits,95),row.p95_wait_min,"raw P95")
            area=10*z["unserved_count"][:-1].sum()
            close(area,row.waiting_area_min,"independent queue integral")
            close(area/row.arrivals_count,row.restricted_mean_wait_min,"restricted wait")
            assert area>=waits.sum() and z["unserved_count"][-1]==row.unserved_count
            close(z["movement_cost"].sum(),row.movement_cost,"total movement")
            close(z["reward"].mean(),row.mean_reward,"reward")
            for field in ["candidate_infeasible","executable_infeasible","repaired"]:
                csvfield={"repaired":"repair_epochs"}.get(field,field+"_epochs")
                assert int(z[field].sum())==getattr(row,csvfield)
            raw_samples+=len(waits);trace_count+=1
        if trace_count%1890==0:print(f"Audited {trace_count}/{len(episodes)} PPO traces",flush=True)
    # The common predictor is a deterministic function of the same causal inflow.
    for _,group in episodes.groupby(["predictor","scenario","environment_seed"]):
        for h in [1,2,4,6]:
            close(group[f"forecast_mae_h{h}"].values,group[f"forecast_mae_h{h}"].iloc[0],"paired frozen forecast")
    per=[]
    for key,g in episodes.groupby(["predictor","method","training_seed","budget_steps","scenario"]):
        per.append(dict(zip(["predictor","method","training_seed","budget_steps","scenario"],key),
                        restricted_mean_wait_min=float(g.waiting_area_min.sum()/g.arrivals_count.sum()),
                        unserved_pct=float(100*g.unserved_count.sum()/g.arrivals_count.sum()),
                        movement_cost=float(g.movement_cost.mean())))
    per=pd.DataFrame(per)
    contrasts=[]
    for surge in [False,True]:
        names=[s[0] for s in scenarios() if s[0]!="legacy" and s[2]==surge]
        chosen=per[(per.predictor=="edge_stgru")&(per.budget_steps==43200)&per.scenario.isin(names)]
        means=chosen.groupby(["method","training_seed"])["restricted_mean_wait_min"].mean().unstack(0)
        if {"F1","F4"}.issubset(means.columns):
            differences=(means.F4-means.F1).dropna()
            contrasts.append(dict(condition="surge" if surge else "new_nominal",n_policy_seeds=len(differences),
                                  mean_difference=float(differences.mean()),paired_differences=differences.tolist()))
    out=V3/"audits";out.mkdir(exist_ok=True)
    suffix="_partial" if partial else ""
    per.to_csv(out/f"policy_per_seed_scenario_independent{suffix}.csv",index=False)
    report=dict(status="PARTIAL_PASS" if partial else "PASS",trained_checkpoints=trained,
                unchanged_first14400_validation_scores=budget_prefix_comparisons,
                evaluation_shards=len(paths),episodes=len(episodes),trace_files=trace_count,
                raw_completed_request_samples=raw_samples,primary_contrasts=contrasts,
                checks=["v2 and supplemental source provenance","all checkpoint identities and validation selection",
                        "same seeded initial14400 training trajectory validation scores","complete per-shard scenario/day matrix",
                        "generated request denominators","raw wait mean/P95/sum","queue integral includes terminal censored waiting",
                        "exact movement and reward aggregation","zero executable resource flags","matched frozen forecast diagnostics",
                        "pool days within scenario then equally weight generator seeds; training seed remains inferential unit"],
                elapsed_seconds=time.perf_counter()-start)
    write_json(out/f"policy_independent_audit{suffix}.json",report);print(json.dumps(report,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--partial",action="store_true");args=p.parse_args();audit(args.partial)
