(() => {
  'use strict';
  const clean = (s, n=160) => String(s || '').slice(0,n);
  const started = performance.now(), pending = new Set();
  let emitted = 0, dropped = 0, ready = false;
  function snapshot() {
    const forms = Array.from(document.forms), all = Array.from(document.querySelectorAll('input,textarea'));
    const eligible = all.filter(e => e.tagName === 'TEXTAREA' || ['text','email','password','tel','number'].includes(e.type));
    return {form_count:forms.length, entry_control_count:eligible.length,
      fields:eligible.slice(0,60).map(e => {
        const style = getComputedStyle(e);
        return {dom_index:all.indexOf(e), type:e.type || 'textarea',
          attributes:Object.fromEntries(['id','name','placeholder','autocomplete','aria-label','title'].map(k=>[k,clean(e.getAttribute(k))])),
          label_texts:Array.from(e.labels || []).slice(0,4).map(l=>clean(l.textContent)),
          aria_labelledby_texts:(e.getAttribute('aria-labelledby') || '').split(/\s+/).slice(0,4).map(id=>clean(document.getElementById(id)?.textContent)),
          form_index:e.form ? forms.indexOf(e.form):null, disabled:!!e.disabled,
          rendered_visible:!!e.getClientRects().length && style.visibility!=='hidden' && style.display!=='none'};
      }),
      forms:forms.slice(0,30).map((f,index)=>({index, method:f.method, action:clean(f.action,8192)})),
      storage_cut:eligible.length>60 || forms.length>30, trace_events_dropped:dropped};
  }
  function emit(kind, data={}) {
    if (emitted++ >= 300) { dropped++; return; }
    const promise = window.__behaviorRecord(JSON.stringify({kind, document_ms:performance.now()-started,...data})).catch(()=>{});
    pending.add(promise); promise.finally(()=>pending.delete(promise));
  }
  Object.defineProperty(window,'__behaviorProbe',{value:{snapshot,flush:()=>Promise.all(Array.from(pending))},configurable:false});
  document.addEventListener('DOMContentLoaded',()=>{ready=true;emit('dom_ready',{snapshot:snapshot()});},{once:true});
  new MutationObserver(records=>{
    if (!ready) return;
    let forms=0, controls=0, attributes=0;
    for (const r of records) {
      if (r.type==='attributes') attributes++;
      for (const n of r.addedNodes || []) {
        if (!(n instanceof Element)) continue;
        forms+=Number(n.matches('form'))+n.querySelectorAll('form').length;
        controls+=Number(n.matches('input,textarea'))+n.querySelectorAll('input,textarea').length;
      }
    }
    if(forms||controls||attributes)emit('dom_change',{forms_added:forms,controls_added:controls,attributes_changed:attributes,snapshot:snapshot()});
  }).observe(document,{subtree:true,childList:true,attributes:true,attributeFilter:['type','action','method','form','aria-label','placeholder']});
  document.addEventListener('submit',e=>emit('submit_event'),true);
  for (const kind of ['input','change']) document.addEventListener(kind,e=>{
    if(e.target instanceof HTMLInputElement||e.target instanceof HTMLTextAreaElement)emit(kind);
  },true);
  const submit=HTMLFormElement.prototype.submit;
  HTMLFormElement.prototype.submit=function(...args){emit('direct_submit');return Reflect.apply(submit,this,args);};
  const nativeFetch=window.fetch;
  window.fetch=function(...args){emit('fetch_call');return Reflect.apply(nativeFetch,this,args);};
  const send=XMLHttpRequest.prototype.send;
  XMLHttpRequest.prototype.send=function(...args){emit('xhr_call');return Reflect.apply(send,this,args);};
  const beacon=navigator.sendBeacon;
  navigator.sendBeacon=function(...args){emit('beacon_call');return Reflect.apply(beacon,this,args);};
  const atob=window.atob;
  window.atob=function(...args){emit('base64_decode_call');return Reflect.apply(atob,this,args);};
  const timer=window.setTimeout;
  window.setTimeout=function(handler,delay,...args){
    if(typeof handler!=='function')return Reflect.apply(timer,this,[handler,delay,...args]);
    return Reflect.apply(timer,this,[function(...passed){emit('timer_fired',{delay_ms:Number(delay)||0});return Reflect.apply(handler,this,passed);},delay,...args]);
  };
})();
