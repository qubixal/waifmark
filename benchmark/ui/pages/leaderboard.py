from __future__ import annotations

import json as _json
import logging
import re as _re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st

from ui.utils import (
    find_run_path,
    flatten_run_rows,
    read_json,
    read_jsonl,
    rows_to_csv_bytes,
    summarize_run,
    write_json,
    write_jsonl,
)

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

LOGGER = logging.getLogger(__name__)


def _short_model_name(model_name: str) -> str:
    if not model_name:
        return "-"
    parts = model_name.split("/")
    if len(parts) >= 2:
        return "/".join(parts[-2:])
    return model_name


def _chart_model_name(model_name: str) -> str:
    """Extract a clean display name from a model path for chart labels.

    Examples:
        models/unsloth__Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf -> Qwen3.5 9B Q4
        models/bartowski__DeepSeek-R1-0528-GGUF/DeepSeek-R1-0528-Qwen3-8B-Q4_K_M.gguf -> DeepSeek R1 0528 Qwen3 8B Q4
    """
    fname = model_name.rsplit("/", 1)[-1]  # take last path segment
    fname = fname.removesuffix(".gguf").removesuffix(".gguf.json")
    # Split on first hyphen after a quantization marker (Q followed by digit)
    m = _re.split(r"-(Q\d)", fname, maxsplit=1)
    if len(m) == 3:
        base = m[0].replace("_", " ").replace("-", " ").strip()
        quant = (m[1] + m[2].split("_")[0]).replace("_", " ").strip()
        return f"{base} {quant}"
    return fname.replace("_", " ").replace("-", " ").strip()


def _format_run_date(run_id: str, run_path: Path) -> str:
    if run_id.startswith("run_"):
        stamp = run_id.removeprefix("run_")
        for fmt in ("%Y%m%dT%H%M%SZ%f", "%Y%m%dT%H%M%SZ"):
            try:
                dt = datetime.strptime(stamp, fmt)
                return dt.strftime("%Y-%m-%d %H:%M UTC")
            except ValueError:
                continue
    try:
        return datetime.fromtimestamp(run_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    except (OSError, FileNotFoundError):
        return "unknown"


def _force_audit(results_dir: Path, run_id: str, task_type: str, task_id: str) -> bool:
    run_path = find_run_path(results_dir, run_id)
    if run_path is None:
        return False

    # Update JSONL first
    jsonl_path = results_dir / f"{task_type}_items.jsonl"
    rows = read_jsonl(jsonl_path)
    updated_jsonl = False
    for row in rows:
        if row.get("run_id") != run_id or row.get("task_id") != task_id:
            continue
        row["requires_review"] = True
        row["forced_review"] = True
        updated_jsonl = True
        break
    if updated_jsonl:
        write_jsonl(jsonl_path, rows)

    # Only update run JSON if JSONL was updated
    if not updated_jsonl:
        return False

    run_payload = read_json(run_path, {})
    for task in run_payload.get(task_type, []):
        if task.get("task_id") != task_id:
            continue
        task["requires_review"] = True
        task["forced_review"] = True
        break
    else:
        return False
    write_json(run_path, run_payload)
    return True


def _build_scatter_chart(chart_data_json: str, org_meta_json: str) -> str:
    html = r"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
@import url('https://fonts.googleapis.com/css2?family=Zen+Maru+Gothic:wght@400;500;700;900&display=swap');
:root {
  --bg: #0F0F1A; --p1: #CF8AB8; --p2: #D158B4; --p3: #F15CCA;
  --font: 'Zen Maru Gothic', sans-serif;
}
* { margin:0; padding:0; box-sizing:border-box; }
html, body { height: 100%; }
body { background:var(--bg); color:#e0e0e0; font-family:var(--font); padding:0; overflow:hidden; }
.chart-wrapper {
  width:100%; height:100%; border-radius:10px; padding:16px 24px 8px;
  box-shadow: 0 0 80px rgba(209,88,180,0.06), 0 4px 40px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.04);
  position:relative;
  display:flex; flex-direction:column;
}
.chart-wrapper::before {
  content:''; position:absolute; inset:0; border-radius:16px; padding:1px;
  background:linear-gradient(160deg, rgba(207,138,184,0.25), rgba(241,92,202,0.05));
  -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
  mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
  -webkit-mask-composite: xor; mask-composite: exclude; pointer-events:none;
}
h1 {
  text-align:left; font-size:22px; font-weight:900; letter-spacing:0.5px;
  background:linear-gradient(90deg, var(--p1) 0%, var(--p2) 70%, var(--p3) 100%);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; margin-bottom:2px;
  flex-shrink:0;
}
.subtitle { text-align:left; font-size:13px; font-weight:500; color:rgba(207,138,184,0.45); margin-bottom:4px; letter-spacing:1.5px; text-transform:uppercase; flex-shrink:0; }
.canvas-container { position:relative; width:100%; flex:1; min-height:200px; }
.legend-container { display:flex; flex-wrap:wrap; justify-content:center; gap:14px; margin-top:4px; padding-top:10px; border-top:2px solid rgba(207,138,184,0.1); flex-shrink:0; }
.legend-item {
  display:flex; align-items:center; gap:6px; font-size:13px; font-weight:500; color:rgba(255,255,255,0.55);
  cursor:pointer; padding:4px 10px; border-radius:20px; border:1px solid transparent; transition:all 0.3s ease; user-select:none;
}
.legend-item:hover { background:rgba(207,138,184,0.08); border-color:rgba(207,138,184,0.15); }
.legend-item.active { background:rgba(207,138,184,0.12); border-color:rgba(207,138,184,0.3); color:rgba(255,255,255,0.85); }
.legend-item.dimmed { opacity:0.35; }
.legend-dot { width:9px; height:9px; border-radius:50%; flex-shrink:0; box-shadow:0 0 6px currentColor; transition:all 0.3s ease; }
.legend-item.dimmed .legend-dot { box-shadow:none; }
</style>
</head><body>
<div class="chart-wrapper">
  <h1>Score vs Latency</h1>
  <div class="subtitle">Interactive Benchmark Chart</div>
  <div class="canvas-container"><canvas id="benchChart"></canvas></div>
  <div class="legend-container" id="legend"></div>
</div>
<script>
const rawData = __CHART_DATA__;
const orgMeta = __ORG_META__;

const GRAY_POINT='#474C60', GRAY_LINE='rgba(58,58,61,0.35)', GRAY_LABEL='rgba(71,67,114,0.45)';
let selectedOrgs=[];

const orgGroups={};
rawData.forEach(d=>{ if(!orgGroups[d.org]) orgGroups[d.org]=[]; orgGroups[d.org].push(d); });
Object.values(orgGroups).forEach(arr=>arr.sort((a,b)=>a.score-b.score));

const datasets=[];
Object.entries(orgGroups).forEach(([org,models])=>{
  const meta=orgMeta[org];
  if(models.length>1){
    datasets.push({
      label:`_line_${org}`, data:models.map(m=>({x:m.score,y:m.time})),
      borderColor:meta.color+"88", borderWidth:2, borderDash:[6,4],
      pointRadius:0, pointHoverRadius:0, showLine:true, fill:false, tension:0, order:2,
      _org:org, _origBorderColor:meta.color+"88"
    });
  }
  datasets.push({
    label:meta.label,
    data:models.map(m=>({x:m.score,y:m.time,modelName:m.model,runId:m.run_id})),
    backgroundColor:meta.color, borderColor:meta.color,
    pointRadius:7, pointHoverRadius:10, pointBorderWidth:2, pointBorderColor:meta.color+'44',
    showLine:false, order:1, _org:org, _origColor:meta.color, _origBorderColor:meta.color+'44'
  });
});

function applyFilter(chart){
  chart.data.datasets.forEach(ds=>{
    const org=ds._org, isLine=ds.label.startsWith('_line_');
    const isActive=selectedOrgs.length===0||selectedOrgs.includes(org);
    if(isLine){ ds.borderColor=isActive?ds._origBorderColor:GRAY_LINE; ds.borderWidth=isActive?2:1; }
    else {
      ds.backgroundColor=isActive?ds._origColor:GRAY_POINT;
      ds.borderColor=isActive?ds._origColor:GRAY_POINT;
      ds.pointBorderColor=isActive?ds._origColor+'44':'transparent';
      ds.pointRadius=isActive?7:5;
    }
  });
  chart.update('none');
}

const labelPlugin={
  id:'pointLabels',
  afterDatasetsDraw(chart){
    const ctx=chart.ctx; ctx.save();
    ctx.font='500 12px "Zen Maru Gothic",sans-serif';
    const labels=[];
    chart.data.datasets.forEach((ds)=>{
      if(ds.label.startsWith('_line_')) return;
      const dsIdx=chart.data.datasets.indexOf(ds);
      const meta=chart.getDatasetMeta(dsIdx);
      const isActive=selectedOrgs.length===0||selectedOrgs.includes(ds._org);
      const chartMidY=(chart.chartArea.top+chart.chartArea.bottom)/2;
      meta.data.forEach((pt,i)=>{
        const item=ds.data[i]; if(!item.modelName) return;
        if(isActive){
          ctx.beginPath();
          const grd=ctx.createRadialGradient(pt.x,pt.y,0,pt.x,pt.y,14);
          grd.addColorStop(0,ds._origColor+'40'); grd.addColorStop(1,ds._origColor+'00');
          ctx.fillStyle=grd; ctx.arc(pt.x,pt.y,14,0,Math.PI*2); ctx.fill();
        }
        const tw=ctx.measureText(item.modelName).width, th=13;
        const above=pt.y<chartMidY;
        labels.push({
          text:item.modelName, ptX:pt.x, ptY:pt.y,
          x:pt.x+10, y:above?pt.y+5:pt.y-5-th, w:tw, h:th,
          color:isActive?ds._origColor+'cc':GRAY_LABEL,
          leaderColor:isActive?ds._origColor+'66':'rgba(71,67,114,0.25)'
        });
      });
    });
    const CANDIDATES=[
      {angle:0,align:'left',baseline:'middle',pref:0},
      {angle:45,align:'left',baseline:'bottom',pref:1},
      {angle:315,align:'left',baseline:'top',pref:1},
      {angle:90,align:'center',baseline:'bottom',pref:2},
      {angle:270,align:'center',baseline:'top',pref:2},
      {angle:135,align:'right',baseline:'bottom',pref:3},
      {angle:225,align:'right',baseline:'top',pref:3},
      {angle:180,align:'right',baseline:'middle',pref:4},
    ];
    const RADIUS=12, PAD=4, PREF_WT=150;
    const pointObstacles=labels.map(l=>({x:l.ptX,y:l.ptY,r:8}));
    const placed=[];
    [...labels].sort((a,b)=>a.ptX-b.ptX).forEach(label=>{
      let bestScore=Infinity, bestPos=null;
      for(const cand of CANDIDATES){
        const rad=cand.angle*Math.PI/180;
        const ax=label.ptX+Math.cos(rad)*RADIUS;
        const ay=label.ptY-Math.sin(rad)*RADIUS;
        const bx=cand.align==='left'?ax:cand.align==='right'?ax-label.w:ax-label.w/2;
        const by=cand.baseline==='top'?ay:cand.baseline==='bottom'?ay-label.h:ay-label.h/2;
        let score=cand.pref*PREF_WT;
        for(const p of placed){
          const ox=Math.min(bx+label.w,p.x+p.w)-Math.max(bx,p.x)+PAD;
          const oy=Math.min(by+label.h,p.y+p.h)-Math.max(by,p.y)+PAD;
          if(ox>0&&oy>0) score+=ox*oy*8;
        }
        for(const po of pointObstacles){
          if(po.x===label.ptX&&po.y===label.ptY) continue;
          const nearX=Math.max(bx,Math.min(po.x,bx+label.w));
          const nearY=Math.max(by,Math.min(po.y,by+label.h));
          const dist=Math.hypot(nearX-po.x,nearY-po.y);
          if(dist<po.r) score+=(po.r-dist)*80;
        }
        if(score<bestScore){ bestScore=score; bestPos={ax,ay,bx,by,align:cand.align,baseline:cand.baseline}; }
      }
      label.ax=bestPos.ax; label.ay=bestPos.ay; label.x=bestPos.bx; label.y=bestPos.by;
      label.align=bestPos.align; label.baseline=bestPos.baseline;
      placed.push({x:bestPos.bx,y:bestPos.by,w:label.w,h:label.h});
    });
    labels.forEach(label=>{
      ctx.fillStyle=label.color; ctx.textAlign=label.align; ctx.textBaseline=label.baseline;
      ctx.fillText(label.text,label.ax,label.ay);
    });
    ctx.restore();
  }
};

const gridGlowPlugin={
  id:'gridGlow',
  beforeDraw(chart){
    const ctx=chart.ctx, area=chart.chartArea;
    const grd=ctx.createLinearGradient(0,area.top,0,area.bottom);
    grd.addColorStop(0,'rgba(207,138,184,0.01)'); grd.addColorStop(1,'rgba(241,92,202,0.02)');
    ctx.fillStyle=grd; ctx.fillRect(area.left,area.top,area.right-area.left,area.bottom-area.top);
  }
};

const FONT='"Zen Maru Gothic",sans-serif';
const ctx=document.getElementById('benchChart').getContext('2d');
const chart=new Chart(ctx,{
  type:'scatter',
  data:{datasets},
  options:{
    responsive:true, maintainAspectRatio:false,
    animation:{duration:350,easing:'easeOutQuart'},
    layout:{padding:{top:10,right:90,bottom:8,left:8}},
    scales:{
      x:{
        type:'linear',
        title:{display:true,text:'SCORE (%)',color:'#CF8AB8',font:{family:FONT,size:13,weight:'700'},padding:{top:10}},
        min:0, max:100,
        grid:{color:'rgba(207,138,184,0.07)',lineWidth:1},
        border:{color:'rgba(207,138,184,0.15)'},
        ticks:{color:'rgba(207,138,184,0.6)',font:{family:FONT,size:12,weight:'500'},callback:v=>v+'%',stepSize:10}
      },
      y:{
        type:'logarithmic',
        title:{display:true,text:'TIME / RESPONSE (s)',color:'#CF8AB8',font:{family:FONT,size:13,weight:'700'},padding:{bottom:10}},
        min:1, max:80,
        grid:{color:'rgba(207,138,184,0.07)',lineWidth:1},
        border:{color:'rgba(207,138,184,0.15)'},
        afterBuildTicks:(axis)=>{
          const ticks=[1,2,3,4,5,6,7,8,9,10,15,20,30,40,50,60,70,80];
          axis.ticks=ticks.map(v=>({value:v}));
        },
        ticks:{
          color:'rgba(207,138,184,0.6)',font:{family:FONT,size:12,weight:'500'},
          callback:function(value){
            const show=[1,2,5,10,20,40,80];
            return show.includes(value)?value+'s':'';
          }
        }
      }
    },
    plugins:{
      legend:{display:false},
      tooltip:{
        backgroundColor:'rgba(15,15,26,0.95)', borderColor:'rgba(207,138,184,0.3)', borderWidth:1,
        titleFont:{family:FONT,size:12,weight:'700'}, bodyFont:{family:FONT,size:11},
        titleColor:'#F15CCA', bodyColor:'#CF8AB8', padding:12, cornerRadius:8,
        filter:(item)=>!item.dataset.label.startsWith('_line_'),
        callbacks:{
          title:(items)=>items[0].raw.modelName||'',
          label:(item)=>`Score: ${item.raw.x}%  ·  Time: ${item.raw.y.toFixed(2)}s`
        }
      }
    }
  },
  plugins:[gridGlowPlugin,labelPlugin]
});

const legendEl=document.getElementById('legend');
const legendItems={};
Object.entries(orgMeta).forEach(([org,meta])=>{
  const div=document.createElement('div');
  div.className='legend-item';
  div.innerHTML=`<span class="legend-dot" style="background:${meta.color};color:${meta.color}"></span>${meta.label}`;
  div.addEventListener('click',()=>{
    if(selectedOrgs.includes(org)){
      selectedOrgs=selectedOrgs.filter(o=>o!==org);
      if(selectedOrgs.length===0) Object.values(legendItems).forEach(el=>el.classList.remove('active','dimmed'));
      else Object.entries(legendItems).forEach(([k,el])=>{ el.classList.toggle('active',selectedOrgs.includes(k)); el.classList.toggle('dimmed',!selectedOrgs.includes(k)); });
    } else {
      if(selectedOrgs.length>=2) selectedOrgs.shift();
      selectedOrgs.push(org);
      Object.entries(legendItems).forEach(([k,el])=>{ el.classList.toggle('active',selectedOrgs.includes(k)); el.classList.toggle('dimmed',selectedOrgs.length>0&&!selectedOrgs.includes(k)); });
    }
    applyFilter(chart);
  });
  legendItems[org]=div;
  legendEl.appendChild(div);
});
</script></body></html>"""
    return html.replace("__CHART_DATA__", chart_data_json).replace("__ORG_META__", org_meta_json)


def render(config: dict[str, Any]) -> None:
    results_dir = BASE_DIR / config["run"]["output_dir"]
    run_files = sorted(results_dir.glob("run_*.json"), reverse=True)

    if not run_files:
        st.info("No benchmark runs found.")
        return

    status = st.session_state.get("results_force_audit_status")
    if status:
        if status["success"]:
            st.success(status["message"])
        else:
            st.error(status["message"])
        st.session_state.results_force_audit_status = None

    st.markdown(
        '<div class="waifmark-page-header">'
        '<div class="waifmark-page-title">Benchmark Leaderboard</div>'
        '<div class="waifmark-page-subtitle">Scoreboard</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    _ORG_META = {
        "QWEN": {"color": "#F15CCA", "label": "Qwen"},
        "GOOG": {"color": "#6BCB77", "label": "Google"},
        "DPSK": {"color": "#3B45FF", "label": "DeepSeek"},
        "MIST": {"color": "#EC7A55", "label": "Mistral"},
        "GLM":  {"color": "#227DFF", "label": "GLM"},
        "ESSE": {"color": "#FECB33", "label": "Essense"},
        "NVDA": {"color": "#83BF0D", "label": "NVIDIA"},
    }

    def _detect_org(model_name: str) -> str:
        low = model_name.lower()
        if "qwen" in low:
            return "QWEN"
        if "gemma" in low:
            return "GOOG"
        if "deepseek" in low or "dpsk" in low:
            return "DPSK"
        if "mistral" in low or "ministral" in low:
            return "MIST"
        if "glm" in low or "chatglm" in low or "4.6v" in low:
            return "GLM"
        if "rnj" in low or "essense" in low:
            return "ESSE"
        if "nemotron" in low:
            return "NVDA"
        return "MISC"

    if "MISC" not in _ORG_META:
        _ORG_META["MISC"] = {"color": "#888888", "label": "Other"}

    leaderboard_rows: list[dict[str, Any]] = []
    chart_data: list[dict[str, Any]] = []
    for path in run_files:
        payload = read_json(path, {})
        summary = summarize_run(payload)
        run_id = payload.get("run_id", path.stem)
        model_name = payload.get("model_name", path.name)
        org = _detect_org(model_name)
        leaderboard_rows.append(
            {
                "Model": _short_model_name(model_name),
                "Overall": round(summary["overall"], 2),
                "Agentic": round(summary["agentic"], 2),
                "Roleplay": round(summary["roleplay"], 2),
                "Date": _format_run_date(run_id, path),
                "# Tasks": len(payload.get("agentic", [])) + len(payload.get("roleplay", [])),
                "_run_id": run_id,
                "_run_file": path.name,
            }
        )
        latency = summary.get("latency", 0.0)
        if latency > 0 and summary["overall"] > 0:
            chart_data.append({
                "model": _chart_model_name(model_name),
                "org": org,
                "score": round(summary["overall"], 2),
                "time": round(latency, 3),
                "run_id": run_id,
            })

    leaderboard_rows.sort(key=lambda row: row["Overall"], reverse=True)

    if chart_data:
        chart_json = _json.dumps(chart_data)
        org_meta_json = _json.dumps(_ORG_META)
        st.components.v1.html(
            _build_scatter_chart(chart_json, org_meta_json),
            height=600,
            scrolling=False,
        )

    import pandas as pd

    leaderboard_df = pd.DataFrame(
        [
            {
                "Model": row["Model"],
                "Overall": row["Overall"],
                "Agentic": row["Agentic"],
                "Roleplay": row["Roleplay"],
                "Date": row["Date"],
                "# Tasks": row["# Tasks"],
            }
            for row in leaderboard_rows
        ]
    )

    top_cols = st.columns([1, 4])
    with top_cols[0]:
        st.download_button(
            "Export CSV",
            data=rows_to_csv_bytes(
                [
                    {
                        "Model": row["Model"],
                        "Overall": row["Overall"],
                        "Agentic": row["Agentic"],
                        "Roleplay": row["Roleplay"],
                        "Date": row["Date"],
                        "# Tasks": row["# Tasks"],
                    }
                    for row in leaderboard_rows
                ]
            ),
            file_name="waifmark_leaderboard.csv",
            mime="text/csv",
            width='stretch',
        )
    with top_cols[1]:
        st.caption("Sortable columns are built into the table. Click a row to open run details.")

    @st.fragment
    def _render_leaderboard_detail() -> None:
        selection = st.dataframe(
            leaderboard_df,
            key="results_leaderboard_table",
            on_select="rerun",
            selection_mode="single-row",
            hide_index=True,
            width='stretch',
            height=min(460, 35 + len(leaderboard_rows) * 35),
        )

        selected_rows = selection.get("selection", {}).get("rows", [])
        if not selected_rows:
            st.info("Select a leaderboard row to inspect task-level detail.")
            return

        selected = leaderboard_rows[selected_rows[0]]
        payload = read_json(results_dir / selected["_run_file"], {})
        rows = flatten_run_rows(payload)

        with st.expander(f"{selected['Model']} · {selected['Date']}", expanded=True):
            st.caption(f"Run: {selected['_run_id']}")
            task_df = pd.DataFrame(
                [
                    {
                        "Module": row["module"],
                        "Task": row["task_id"],
                        "Category": row["category"],
                        "Score": round(float(row.get("score_100", 0.0)), 2),
                        "Review": "Forced" if row.get("forced_review") else "Flagged" if row.get("triage") else "Clean",
                    }
                    for row in rows
                ]
            )
            st.dataframe(task_df, hide_index=True, width='stretch', height=min(320, 35 + len(rows) * 35))

            for row in rows:
                badge = "Forced review" if row.get("forced_review") else "Needs review" if row.get("triage") else "Clean"
                with st.expander(f"{row['module']} · {row['task_id']} · {row['score_100']:.1f} · {badge}"):
                    st.markdown("**Prompt**")
                    st.write(row["prompt"])
                    st.markdown("**Response**")
                    st.write(row["response"])
                    c = st.columns([1, 1, 1, 1.2])
                    with c[0]:
                        st.metric("Rule score", f"{row['rule_score']:.1f}" if isinstance(row.get("rule_score"), (int, float)) else "-")
                    with c[1]:
                        st.metric("Judge score", f"{row['judge_score']:.1f}" if isinstance(row.get("judge_score"), (int, float)) else "-")
                    with c[2]:
                        st.metric("Review", badge)
                    with c[3]:
                        if st.button(
                            "Force Audit",
                            key=f"force_audit_{selected['_run_id']}_{row['module']}_{row['task_id']}",
                            width='stretch',
                        ):
                            success = _force_audit(results_dir, selected["_run_id"], row["module"], row["task_id"])
                            st.session_state.results_force_audit_status = {
                                "success": success,
                                "message": "Review queue updated." if success else "Could not update the run and JSONL review records.",
                            }
                            st.rerun()
                    if row.get("judge_rationale"):
                        st.markdown("**Judge rationale**")
                        st.write(row["judge_rationale"])

    _render_leaderboard_detail()
