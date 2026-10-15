"use strict";
const byId = id => document.getElementById(id);
let mode = "url", lastReport = null;
const basisLabels = {http_source:"HTTP source",browser_rendered:"Browser-rendered DOM (user supplied)",user_supplied_unknown:"Capture method unknown"};
for(const name of ["url","html","demo"]){
  byId(name+"-tab").addEventListener("click",()=>{
    mode=name;
    for(const other of ["url","html","demo"]){
      byId(other+"-panel").hidden=other!==name;
      byId(other+"-tab").classList.toggle("selected",other===name);
      byId(other+"-tab").setAttribute("aria-selected",String(other===name));
    }
  });
}
function text(id,value){byId(id).textContent=value;}
function showReport(r){
  lastReport=r;byId("empty").hidden=true;byId("output").hidden=false;
  const inconclusive=r.status!=="prediction_complete";
  byId("result").classList.toggle("danger",!inconclusive&&r.static_model_prediction==="phish");
  byId("result").classList.toggle("inconclusive",inconclusive);
  text("result-tag",inconclusive?"Needs review":r.needs_review?"Review findings":"Analysis complete");
  text("result-title",r.display_result||"Analysis could not complete");
  text("result-url",r.page_url||"");
  const score=typeof r.phishing_score==="number"&&Number.isFinite(r.phishing_score);
  byId("score-panel").hidden=!score;
  if(score){text("score",r.phishing_score.toFixed(3));byId("score-bar").style.width=(r.phishing_score*100).toFixed(2)+"%";}
  byId("facts").replaceChildren();
  if(r.capture_basis){
    for(const [label,value] of [["Capture",r.behavior?"Observed browser DOM":basisLabels[r.capture_basis]||r.capture_basis],["HTML characters",Number(r.html_characters).toLocaleString()],["Features",r.model?.feature_count||106]]){
      const row=document.createElement("div"),dt=document.createElement("dt"),dd=document.createElement("dd");
      dt.textContent=label;dd.textContent=String(value);row.append(dt,dd);byId("facts").append(row);
    }
  }
  byId("flags").replaceChildren();
  const notes=(r.active_quality_flags||[]).map(n=>n.replaceAll("_"," "));
  if(r.capture_basis==="http_source")notes.unshift("JavaScript-created content was not captured.");
  if(r.capture_basis==="browser_rendered"&&!r.capture_basis_verified)notes.unshift("Browser capture method was supplied by the user and has not been verified.");
  if(r.behavior)notes.unshift("Browser observation finished. Test requests were blocked. Some website features are deliberately restricted.");
  if(!notes.length&&!r.error)notes.push("No extractor quality flags were raised.");
  for(const note of notes){const li=document.createElement("li");li.textContent=note;byId("flags").append(li);}
  byId("flags-panel").hidden=!notes.length;
  byId("error").hidden=!r.error;text("error",r.error||"");
  byId("behavior-panel").hidden=!r.behavior;
  byId("behavior-findings").replaceChildren();
  if(r.behavior){
    const findings=r.behavior.rules.map(rule=>rule.finding);
    const s=r.behavior.signals;
    findings.unshift(`Forms added after loading: ${s.forms_added_after_ready}. Credential controls observed: ${s.maximum_passwordlike_controls}.`);
    if(r.behavior.critical_resource_failures.length)findings.push("Some scripts or required page resources could not be captured.");
    for(const finding of findings){const li=document.createElement("li");li.textContent=finding;byId("behavior-findings").append(li);}
    text("source-score",`Before browser rendering: ${r.source_analysis.phishing_score.toFixed(3)}. After rendering: ${r.phishing_score.toFixed(3)}. Findings request review; they do not automatically change the model label.`);
  }
  text("result-note",inconclusive?"Review the capture before relying on this result.":"This model result does not guarantee a website is safe. Review the browser findings and capture notes.");
  byId("download").hidden=!r.saved_report_filename;
}
byId("analysis-form").addEventListener("submit",async event=>{
  event.preventDefault();
  const payload=mode==="demo"?{mode:"demo",case:byId("demo-case").value}:mode==="url"?{mode:byId("analysis-type").value,url:byId("url").value.trim()}:{mode,page_url:byId("page-url").value.trim(),basis:byId("basis").value,html:byId("html").value};
  if(!((mode==="demo")||(mode==="url"&&payload.url)||(mode==="html"&&payload.page_url&&payload.html.trim()))){showReport({status:"input_required",display_result:"Enter a page to analyze",error:mode==="url"?"Enter a complete public website URL.":"Provide a page URL and its captured HTML."});return;}
  byId("analyze").disabled=true;text("analyze","Analyzing…");
  byId("result").setAttribute("aria-busy","true");text("result-tag","Analysis running");
  try{
    const response=await fetch("/api/analyze",{method:"POST",headers:{"Content-Type":"application/json","X-Detector-Token":document.querySelector('meta[name="detector-token"]').content},body:JSON.stringify(payload)});
    showReport(await response.json());
  }catch(error){showReport({status:"connection_failed",display_result:"Interface connection lost",error:"Keep the Python interface running in your terminal and try again."});}
  finally{byId("analyze").disabled=false;text("analyze","Analyze page ↗");byId("result").setAttribute("aria-busy","false");}
});
byId("download").addEventListener("click",()=>{
  if(!lastReport)return;
  const blob=new Blob([JSON.stringify(lastReport,null,2)+"\n"],{type:"application/json"});
  const href=URL.createObjectURL(blob),a=document.createElement("a");a.href=href;a.download=lastReport.saved_report_filename||"analysis_report.json";a.click();setTimeout(()=>URL.revokeObjectURL(href),1000);
});
fetch("/api/state").then(r=>r.json()).then(r=>text("ready",r.ready?"Model ready":"Not ready")).catch(()=>text("ready","Connection lost"));
