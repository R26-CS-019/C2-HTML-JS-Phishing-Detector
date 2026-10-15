# Connected browser detector

This update connects browser observation to your trained detector and its interface. It watches a page for six seconds, records changes, and scores the rendered page with your existing 106-feature model.

## First run

Copy the ZIP contents directly into `D:\research_new\HTML_JS_PhishingDetector`, keeping the folder layout. It replaces script 29 and some interface/report files, and adds browser modules, scripts 34/35 and one test file. Previous versions of replaced files are included in `behavior_v1_previous_files`. No fitted model, original extractor, training data or evaluation result is replaced.

In your existing project terminal:

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\requirements_behavior.txt
.\.venv\Scripts\python.exe -m playwright install chromium
.\.venv\Scripts\python.exe .\34_verify_integrated_detector.py
```

Use the new browser requirements for this update. Keep the frozen ML package versions. The older runtime requirements pin an older browser package.

Expected final result: **Integrated checks: 9/9**. These built-in examples visit no live website. The report is saved under `reports/browser_pipeline_v1`.

Start the usual interface:

```powershell
.\.venv\Scripts\python.exe .\30_start_detector_ui.py --threads 4
```

Open the printed address. Choose **Try a demo**, keep **Login form appears after a delay**, then press **Analyze page**. You should see one form added after loading. Other examples cover an encoded script, an external script, ordinary login/contact forms and a redirect.

Toy examples check observation and integration, not classification accuracy. The frozen model can flag an ordinary toy login or contact page. Passing these checks does not mean those model labels are correct.

## What happens during analysis

The browser records new forms, credential controls, redirects, external scripts, Base64 decoding, fetch/XHR/beacon activity and attempted use of a synthetic test password. The interface shows the model score before and after rendering, plus these findings.

The model still uses exactly 106 features, float32 input, best iteration 1269 and threshold 0.50. Browser mode feeds it the passive rendered main-frame HTML. JavaScript-created content can therefore reach the existing feature extractor. Observed browser signals are separate evidence, not extra model inputs.

Behavior rules request review. A credential form, encoding or cross-origin login alone does not prove phishing or override the frozen model label. Incomplete capture withholds an evaluated prediction while keeping diagnostic scores in the report.

The gateway allows GET requests to validated public IP addresses on ports 80/443, pins connections and checks TLS certificates. It forwards no browser cookies, authentication headers or request bodies. Before entering test values, it blocks every outgoing request. Service workers, WebSockets, popups, downloads, images, media and fonts are restricted. Each analysis uses a fresh browser context and a bounded worker process.

Some pages need blocked resources or POST API calls to render. Required document/script/style/API failures and truncated observation produce an inconclusive result. Browser startup failure also stays a failure; there is no silent source-only fallback.

For suspected phishing pages, run the browser component in an isolated disposable research VM. Chromium sandboxing and request interception are useful controls; this is not a hardened malware-analysis environment.

## Interface and command line

**Website URL** offers browser analysis or the original source-only method. Saved HTML analysis remains available.

```powershell
.\.venv\Scripts\python.exe .\29_run_webpage_detector.py --demo delayed_login
.\.venv\Scripts\python.exe .\29_run_webpage_detector.py --browser-url "https://example.com"
```

Original `--url` continues to capture source without executing JavaScript.

## Evaluate the connected system

Script 35 accepts the same four CSV columns as step 32: `url,label,label_source,label_checked_at`. Use a separately selected, independently labelled list.

```powershell
.\.venv\Scripts\python.exe .\35_evaluate_browser_urls.py --urls ".\inputs\fresh_browser_test_urls.csv" --check-list
.\.venv\Scripts\python.exe .\35_evaluate_browser_urls.py --urls ".\inputs\fresh_browser_test_urls.csv"
```

Default selection: 50 benign + 50 phish. A separately planned balanced sample can use `--per-class N`.

New reports go under `reports/browser_url_evaluation_v1`. The transactional checkpoint preserves one attempt per URL. Completed URLs are not revisited on resume. An interrupted capture stays recorded separately.

Report coverage and per-class coverage with accuracy, precision, recall, F1 and false-positive rate. Failures, interruptions and incomplete captures are excluded from classification metrics and recorded separately. Labels come from independent evidence, not the model. Do not replace failed URLs or relabel disagreements to improve a result.

The old URL pilot has already been reviewed during development. Reusing it is a development rerun, not a fresh final holdout. Preserve its original 89.66% accuracy on 58 scored pages and 58% coverage as a separate source-only result.

Complete analysis dictionaries, main HTML hashes and resource hashes are saved in the checkpoint. Downloaded HTML/resource bodies are not archived by this evaluator.

The report also compares source and rendered predictions on the same browser attempts where both predictions are complete. It reports that paired subset's size and separate metrics. This conditional comparison does not replace full coverage statistics or the old source-only pilot.

## Shared backend integration

```python
from detector_runtime.model import FrozenDetector
from detector_runtime.behavior import analyze_browser_url, analyze_browser_capture

detector = FrozenDetector(project_directory, threads=4)
result = analyze_browser_url(detector, public_url)
```

Run this function on the isolated research host. It returns a JSON-compatible result. A backend can display `display_result`, `phishing_score`, `evaluated_prediction`, `behavior.rules`, `feature_changes` and capture status. The loopback UI is a local tool, not a public service.

For captured resources, use `analyze_browser_capture(detector, final_url, resources)`. Each absolute resource URL maps to `{"headers":{"content-type":"..."},"body":"captured text","status":200}`. Missing resources are blocked and flagged. Replay never falls back to the live network. The caller owns capture provenance and ground truth.

## Verification and remaining work

The actual saved model was used in nine connected observation/model checks, all passing. Fourteen focused decision/network/UI/evaluation checks and sixteen original runtime tests passed. The real interface displayed the delayed form finding and saved its report. Windows worker-job cleanup also passed.

The agent's restricted Windows environment could not launch Chromium with normal renderer sandboxing. Trusted offline fixture QA used a test-only launch substitution inside the existing restricted environment. That substitution is not shipped. Delivered browser capture explicitly requests Chromium sandboxing and has no fallback that disables it.

**Script 34 must pass in your normal project terminal before calling the installed browser pipeline verified on your computer.** All nine observation/model checks also passed with Playwright 1.63 and its matching Chromium 153 browser, using the same trusted-fixture-only QA substitution. Normal sandboxed startup remains unverified in this restricted environment and requires the local check. See `BEHAVIOR_VERIFICATION.json` for the exact limitations.

The software connection has been implemented. Full research validation still needs a fresh labelled browser evaluation, error analysis, comparison with source capture, and the actual group website connection. A separately trained behavior classifier or automatic rule override would need separate development and validation; none is claimed here.

Finite waits and selected interactions can miss behavior. Embedded-frame HTML is outside the main-frame model input. JavaScript hooks can be evaded or tampered with. These checks do not establish dynamic phishing accuracy or resistance to every obfuscation technique.

Official references: [Chromium launch and sandbox](https://playwright.dev/python/docs/api/class-browsertype#browser-type-launch), [request routing](https://playwright.dev/python/docs/api/class-browsercontext#browser-context-route), [WebSocket routing](https://playwright.dev/python/docs/api/class-websocketroute), [Playwright release](https://pypi.org/project/playwright/).
