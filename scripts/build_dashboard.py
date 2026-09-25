#!/usr/bin/env python3
"""Build a compact, filterable HTML research dashboard from Qlib Recorder."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from mlflow.tracking import MlflowClient

from experiment_tracking import EXPERIMENT_NAME


def load_runs(tracking_dir: Path) -> list[dict]:
    db_path = (tracking_dir / "mlflow.db").resolve()
    if not db_path.exists():
        raise FileNotFoundError(f"experiment database not found: {db_path}; run/import experiments first")
    client = MlflowClient(tracking_uri=f"sqlite:///{db_path}")
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(f"experiment {EXPERIMENT_NAME!r} not found")
    runs = client.search_runs(
        [experiment.experiment_id], max_results=50_000, order_by=["attributes.start_time DESC"]
    )
    output = []
    for run in runs:
        tags = run.data.tags
        output.append({
            "name": run.info.run_name or run.info.run_id,
            "run_id": run.info.run_id,
            "started_at": datetime.fromtimestamp(run.info.start_time / 1000, timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M"),
            "status": run.info.status,
            "phase": tags.get("phase", run.data.params.get("phase", "?")),
            "model": tags.get("model", run.data.params.get("model", "?")),
            "transform": tags.get("input_transform", run.data.params.get("input_transform", "?")),
            "legacy": tags.get("result_status") == "legacy_import",
            "metrics": run.data.metrics,
            # Keep local filesystem paths out of the shareable HTML snapshot.
            "params": {
                key: run.data.params[key]
                for key in ("model_recipe", "selected_lightgbm_rounds")
                if key in run.data.params
            },
        })
    return output


HTML = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>FinTechathon QR | 研究看板</title>
<style>
:root{--ink:#17202a;--muted:#627080;--line:#e4e9ef;--bg:#f5f7fa;--card:#fff;--blue:#2457c5;--green:#17845b;--red:#c04450;--navy:#13233c}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif}
header{background:var(--navy);color:#fff;padding:25px max(24px,calc((100vw - 1180px)/2)) 22px}header h1{font-size:22px;margin:0 0 4px}header p{margin:0;color:#c7d1df}.wrap{max-width:1180px;margin:22px auto;padding:0 18px}
.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.card,.panel{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:17px}.card .label{color:var(--muted);font-size:12px}.card .value{font-size:23px;font-weight:700;margin-top:5px}.card .note{font-size:12px;color:var(--muted);margin-top:1px}
.panel{margin-top:14px}.panel h2{font-size:16px;margin:0 0 12px}.sub{color:var(--muted);font-size:12px;margin:-7px 0 12px}
.filters{display:flex;align-items:end;gap:10px;flex-wrap:wrap}.filters label{display:grid;gap:4px;color:var(--muted);font-size:12px}select{min-width:135px;padding:8px 10px;border:1px solid var(--line);border-radius:6px;background:white;color:var(--ink)}button{padding:8px 12px;border:1px solid var(--line);border-radius:6px;background:white;cursor:pointer;color:var(--ink)}button:hover{border-color:var(--blue);color:var(--blue)}
.split{display:grid;grid-template-columns:1.1fr .9fr;gap:14px}.chart{position:relative;padding:5px 0 0 0}.bar-row{display:grid;grid-template-columns:170px 1fr 72px;gap:10px;align-items:center;margin:9px 0;font-size:12px}.bar-label{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.track{height:15px;background:#f0f3f7;border-radius:3px;position:relative}.zero{position:absolute;top:-3px;bottom:-3px;width:1px;background:#758295}.bar{position:absolute;top:2px;height:11px;border-radius:2px}.bar.pos{background:var(--blue)}.bar.neg{background:var(--red)}.score{text-align:right;font-variant-numeric:tabular-nums}.empty{color:var(--muted);padding:14px 0}
.phase-cards{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.phase{padding:13px;border:1px solid var(--line);border-radius:8px}.phase strong{font-size:13px}.phase .big{font-size:20px;font-weight:700;margin:5px 0}.phase p{font-size:12px;color:var(--muted);margin:0}
.table-wrap{overflow:auto}table{width:100%;border-collapse:collapse;font-size:12px;white-space:nowrap}th,td{text-align:left;border-bottom:1px solid var(--line);padding:9px 8px}th{color:var(--muted);font-weight:600;background:#fafbfc;position:sticky;top:0}td.num{text-align:right;font-variant-numeric:tabular-nums}.pill{display:inline-block;padding:2px 7px;border-radius:10px;background:#eef2f8;color:#41536c;font-size:11px}.best{color:var(--green);font-weight:700}.foot{color:var(--muted);font-size:11px;padding:12px 2px 24px}.actions{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;color:var(--muted);font-size:12px}
@media(max-width:760px){.cards{grid-template-columns:repeat(2,1fr)}.split,.phase-cards{grid-template-columns:1fr}.bar-row{grid-template-columns:120px 1fr 60px}}
</style></head><body>
<header><h1>FinTechathon 赛题五 · 量价预测研究看板</h1><p>固定研究视图：候选比较、冻结检验与运行历史。刷新前先重建看板。</p></header>
<main class="wrap">
<section class="cards" id="kpis"></section>
<section class="panel"><h2>筛选实验</h2><div class="filters">
<label>阶段<select id="phase"><option value="all">全部阶段</option></select></label>
<label>模型<select id="model"><option value="all">全部模型</option></select></label>
<label>输入变换<select id="transform"><option value="all">全部变换</option></select></label>
<button id="reset">重置筛选</button><span style="color:var(--muted);font-size:12px">分数按赛题官方综合评分记录；S 不展示测试成绩。</span>
</div></section>
<div class="split">
<section class="panel"><h2>D 阶段候选综合分</h2><p class="sub">只比较开发段候选；负分以红色显示。筛选器可缩小模型或变换范围。</p><div id="score-chart" class="chart"></div></section>
<section class="panel"><h2>核心指标对照</h2><p class="sub">D/H 有历史标签评分；稳定性项为 1 − Jaccard 距离，不代表真实成交换手。</p><div class="table-wrap"><table><thead><tr><th>阶段 / 方案</th><th>Rank IC</th><th>Top组超额</th><th>稳定性</th><th>综合分</th></tr></thead><tbody id="metric-table"></tbody></table></div></section>
</div>
<section class="panel"><h2>阶段状态</h2><div class="phase-cards" id="phase-cards"></div></section>
<section class="panel"><h2>D 单因子诊断</h2><p class="sub">最新 D 诊断记录；Rank IC 描述单因子横截面排序相关性，不能单独决定最终模型。多因子模型仍按官方综合分比较。</p><div class="table-wrap"><table><thead><tr><th>因子</th><th>覆盖率</th><th>平均 Rank IC</th><th>Rank IC IR</th><th>Rank IC 为正比例</th></tr></thead><tbody id="factor-table"></tbody></table></div></section>
<section class="panel"><div class="actions"><div><h2 style="display:inline">运行记录</h2>　<span id="run-count"></span></div><button id="download">导出当前筛选 CSV</button></div><div class="table-wrap"><table><thead><tr><th>启动时间</th><th>阶段</th><th>模型</th><th>变换</th><th>模型配方</th><th>官方综合分</th><th>Rank IC</th><th>Top组超额</th><th>轮数</th><th>状态</th><th>来源</th><th>Run ID</th></tr></thead><tbody id="runs-table"></tbody></table></div></section>
<p class="foot">本页由本地 Qlib Recorder / MLflow 记录库生成；预测与原始数据留在本机。D/H 结果属于历史评估，不能等同未来收益保证。</p>
</main><script>
const RUNS = __RUNS__;
const fmt=(x,n=4)=>Number.isFinite(Number(x))?Number(x).toFixed(n):"—";
const esc=s=>String(s??"").replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const recipeLabel=r=>{try{const p=JSON.parse(r.params.model_recipe||'{}');const keys=['num_leaves','learning_rate','min_child_samples','reg_lambda','alpha'];const vals=keys.filter(k=>p[k]!==undefined).map(k=>`${k}=${p[k]}`);return vals.join(' · ')||JSON.stringify(p)}catch{return r.params.model_recipe||'—'}};
const options=(id,key)=>{const sel=document.getElementById(id);[...new Set(RUNS.map(r=>r[key]).filter(Boolean))].sort().forEach(v=>{const o=document.createElement('option');o.value=v;o.textContent=v;sel.appendChild(o)})};
options('phase','phase');options('model','model');options('transform','transform');
const metric=(r,k)=>Number(r.metrics[k]);
function filtered(){return RUNS.filter(r=>['phase','model','transform'].every(k=>{const v=document.getElementById(k).value;return v==='all'||r[k]===v}))}
function render(){const rows=filtered(),d=rows.filter(r=>r.phase==='D'&&Number.isFinite(metric(r,'final_score'))).sort((a,b)=>metric(b,'final_score')-metric(a,'final_score'));
document.getElementById('run-count').textContent=`${rows.length} 条运行`;
const selectedD=RUNS.filter(r=>r.phase==='D'&&Number.isFinite(metric(r,'final_score'))).sort((a,b)=>metric(b,'final_score')-metric(a,'final_score'))[0];
const h=RUNS.filter(r=>r.phase==='H').sort((a,b)=>b.started_at.localeCompare(a.started_at))[0];
const s=RUNS.filter(r=>r.phase==='S').sort((a,b)=>b.started_at.localeCompare(a.started_at))[0];
const factorRun=RUNS.filter(r=>r.model==='factor_diagnostic').sort((a,b)=>b.started_at.localeCompare(a.started_at))[0];
document.getElementById('kpis').innerHTML=[
['累计运行',RUNS.length,'含历史结果与新实验'],['D 当前最佳',selectedD?fmt(metric(selectedD,'final_score')):'—',selectedD?`${selectedD.model} · ${selectedD.transform}`:'暂无 D 评分'],['H 冻结方案',h?fmt(metric(h,'final_score')):'—',h?'2024 历史验证':'暂无 H 记录'],['S 预测覆盖',s?Number(s.metrics.prediction_rows||0).toLocaleString():'—',s?'测试集行数；无本地评分':'暂无 S 记录']
].map(x=>`<article class="card"><div class="label">${x[0]}</div><div class="value">${x[1]}</div><div class="note">${x[2]}</div></article>`).join('');
const scores=d.map(r=>metric(r,'final_score')),lo=Math.min(0,...scores),hi=Math.max(0,...scores),range=Math.max(hi-lo,1e-9),zero=(-lo/range)*100;
document.getElementById('score-chart').innerHTML=d.length?d.slice(0,24).map(r=>{const v=metric(r,'final_score'),left=(Math.min(0,v)-lo)/range*100,width=Math.abs(v)/range*100;return `<div class="bar-row"><div class="bar-label" title="${esc(r.name)} · ${esc(r.params.model_recipe||'')}">${esc(r.model)} · ${esc(r.transform)}<br><span style="color:var(--muted)">${esc(recipeLabel(r))}</span></div><div class="track"><i class="zero" style="left:${zero}%"></i><i class="bar ${v>=0?'pos':'neg'}" style="left:${left}%;width:${Math.max(width,.3)}%"></i></div><div class="score">${fmt(v)}</div></div>`}).join(''):'<div class="empty">筛选条件下没有 D 阶段评分。</div>';
const componentRows=[...d.slice(0,10),...rows.filter(r=>r.phase==='H'&&Number.isFinite(metric(r,'final_score'))).slice(0,2)];
document.getElementById('metric-table').innerHTML=componentRows.length?componentRows.map(r=>{const stability=Number.isFinite(metric(r,'mean_turnover'))?1-metric(r,'mean_turnover'):NaN;return `<tr><td>${esc(r.phase)} / ${esc(r.model)}<br><span class="pill">${esc(r.transform)}</span></td><td class="num">${fmt(metric(r,'ic_mean'))}</td><td class="num">${fmt(metric(r,'annual_excess'))}</td><td class="num">${fmt(stability)}</td><td class="num ${r===selectedD?'best':''}">${fmt(metric(r,'final_score'))}</td></tr>`}).join(''):'<tr><td colspan="5" class="empty">暂无可展示评分</td></tr>';
document.getElementById('phase-cards').innerHTML=['D','H','S'].map(p=>{const arr=RUNS.filter(r=>r.phase===p);const last=p==='D'?arr.filter(r=>Number.isFinite(metric(r,'final_score'))).sort((a,b)=>metric(b,'final_score')-metric(a,'final_score'))[0]:arr.sort((a,b)=>b.started_at.localeCompare(a.started_at))[0];let big=p==='S'?(last?`${Number(last.metrics.prediction_rows||0).toLocaleString()} 行预测`:'暂无记录'):(last?`${fmt(metric(last,'final_score'))} 综合分`:'暂无记录');let note=p==='D'?'开发期候选比较；该段承担选型，不是独立最终检验。':p==='H'?'冻结配方的 2024 检验；H 结果不应用于事后调参。':'2025—2026 提交预测；不读取测试标签、不报告测试分数。';return `<div class="phase"><strong>${p} 阶段</strong><div class="big">${big}</div><p>${note}<br>${arr.length} 条运行记录</p></div>`}).join('');
const factorNames=factorRun?Object.keys(factorRun.metrics).filter(k=>k.startsWith('rankic_mean__')).map(k=>k.slice('rankic_mean__'.length)):[];
document.getElementById('factor-table').innerHTML=factorNames.length?factorNames.map(name=>`<tr><td>${esc(name)}</td><td class="num">${fmt(factorRun.metrics['coverage__'+name],3)}</td><td class="num">${fmt(factorRun.metrics['rankic_mean__'+name])}</td><td class="num">${fmt(factorRun.metrics['rankic_ir__'+name])}</td><td class="num">${fmt(factorRun.metrics['rankic_positive_ratio__'+name]*100,1)}%</td></tr>`).join(''):'<tr><td colspan="5" class="empty">暂无 D 因子诊断；运行 scripts/factor_diagnostics.py 后刷新。</td></tr>';
const recent=[...rows].sort((a,b)=>b.started_at.localeCompare(a.started_at)).slice(0,100);
document.getElementById('runs-table').innerHTML=recent.length?recent.map(r=>`<tr><td>${esc(r.started_at)}</td><td><span class="pill">${esc(r.phase)}</span></td><td>${esc(r.model)}</td><td>${esc(r.transform)}</td><td title="${esc(r.params.model_recipe||'')}">${esc(recipeLabel(r))}</td><td class="num">${fmt(metric(r,'final_score'))}</td><td class="num">${fmt(metric(r,'ic_mean'))}</td><td class="num">${fmt(metric(r,'annual_excess'))}</td><td class="num">${fmt(metric(r,'selected_lightgbm_rounds'),0)}</td><td>${esc(r.status)}</td><td>${r.legacy?'历史导入':'实验运行'}</td><td title="${esc(r.run_id)}">${esc(r.run_id.slice(0,10))}…</td></tr>`).join(''):'<tr><td colspan="12" class="empty">当前筛选下无记录。</td></tr>';
}
['phase','model','transform'].forEach(id=>document.getElementById(id).addEventListener('change',render));
document.getElementById('reset').onclick=()=>{['phase','model','transform'].forEach(id=>document.getElementById(id).value='all');render()};
document.getElementById('download').onclick=()=>{const rs=filtered(),headers=['started_at','phase','model','transform','model_recipe','final_score','ic_mean','annual_excess','mean_turnover','selected_lightgbm_rounds','run_id'];const csv=[headers.join(','),...rs.map(r=>headers.map(k=>{let v=k==='model_recipe'?r.params.model_recipe:(k in r?r[k]:(k in r.metrics?r.metrics[k]:''));return `"${String(v??'').replaceAll('"','""')}"`}).join(','))].join('\n');const a=document.createElement('a');a.href=URL.createObjectURL(new Blob(['\ufeff'+csv],{type:'text/csv;charset=utf-8'}));a.download='experiment_runs.csv';a.click();URL.revokeObjectURL(a.href)};
render();
</script></body></html>'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-dir", type=Path, default=Path("outputs/experiment_tracking"))
    parser.add_argument("--output", type=Path, default=Path("outputs/dashboard/index.html"))
    args = parser.parse_args()
    rows = load_runs(args.tracking_dir)
    encoded = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
    encoded = encoded.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(HTML.replace("__RUNS__", encoded), encoding="utf-8")
    print(f"Built dashboard for {len(rows)} runs: {args.output.resolve()}")


if __name__ == "__main__":
    main()
