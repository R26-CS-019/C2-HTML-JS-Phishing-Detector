# Webpage detector runtime, version 1.0

## Browser update installed — 12 October 2026

For the connected browser interface, trusted demos and scripts 34/35, follow
`READ_ME_BEHAVIOR.md`. Its browser setup uses `requirements_behavior.txt`
(Playwright 1.63.0). Keep the existing frozen ML package versions.

The sections below retain the source-capture workflow and earlier controlled
lab instructions. Use the browser update's requirements for the updated system;
`requirements_runtime.txt` records the older lab's Playwright version.

The trained model, its 106 features, threshold 0.50 and existing evaluation
results are preserved. Browser observations accompany the model result.

Local verification on 12 October 2026 passed: 69 software tests, the saved
Step 33 evaluation integrity check, and 9/9 connected browser checks with
Chromium sandboxing enabled. The delayed-form demo also passed through the
actual interface. These are software checks, not new phishing accuracy results.

The installed browser is Chromium's headless shell, which the connected
pipeline uses. To install the optional visible browser for `--headed` lab
sessions, run `python -m playwright install chromium` from the project `.venv`.
Its larger download was interrupted during this setup; the headless pipeline
is installed and verified.

This connects your **already trained and evaluated** `lightgbm_context_all_features` model to a webpage source crawler, an offline HTML analyzer, and a local interface. A separate controlled browser lab checks delayed and encoded JavaScript behavior and compares features before and after rendering.

You do **not** need to train the static model again. The existing balanced training/validation data, saved model, 106 feature definitions, float32 input, best iteration 1269, threshold 0.50, and official-test results remain unchanged.

## 1. Extract into your existing project

Open the ZIP and copy its contents directly into:

```text
D:\research_new\HTML_JS_PhishingDetector
```

After copying, the layout should include:

```text
HTML_JS_PhishingDetector/
  09_extract_static_features.py                  existing, unchanged
  15_extract_form_context.py                    existing, unchanged
  29_run_webpage_detector.py                    new
  30_start_detector_ui.py                       new
  31_test_dynamic_runtime.py                    new
  detector_runtime/                            new folder; copy all its files
  examples/                                    new
  tests/                                       new
  requirements_runtime.txt                     new
  reports/
    static_baseline_freeze_v1/
      static_baseline_freeze_report.json        existing
    features_context_v2/
      static_feature_schema.json                existing
    form_context_training_v2/
      run_config.json                           existing
      lightgbm_context_all_features/
        model_bundle.joblib                     existing
```

The ZIP contains no model or dataset. It does not replace scripts 01–28. Keep your original model and reports. Do not place the new scripts in an extra nested `webpage_detector_v1` folder: their default project directory is the folder containing each script. You can instead pass `--project-dir` explicitly if you prefer another location.

## 2. Verify the real model on your laptop

Open the existing project in VS Code. In its PowerShell terminal run:

```powershell
Set-Location "D:\research_new\HTML_JS_PhishingDetector"
.\.venv\Scripts\python.exe .\29_run_webpage_detector.py --check
```

Expected final message starts with `Frozen runtime verified: 106 features`. This checks the original artifact hashes, frozen schema, dependency versions, model metadata, feature order and best iteration, then runs a small offline inference check. It does not measure accuracy or refit a model.

No new packages are needed for this check or for the URL/HTML interface if your original working `.venv` is intact. The runtime checks the original package versions before loading the serialized model:

| Package | Frozen version |
| --- | --- |
| NumPy | 2.5.3 |
| scikit-learn | 1.9.1 |
| LightGBM | 4.7.0 |
| joblib | 1.6.0 |
| threadpoolctl | 3.7.0 |
| DuckDB | 1.5.6 |
| tldextract | 5.4.0 |

DuckDB remains an import dependency of unchanged script 09; the live detector does not query Parquet. tldextract uses the same offline private-suffix-aware site grouping as training.

If the frozen report was saved elsewhere, pass its path:

```powershell
.\.venv\Scripts\python.exe .\29_run_webpage_detector.py --check --freeze-report "FULL_PATH_TO_YOUR_ORIGINAL_FREEZE_REPORT.json"
```

## 3. Start the interface

```powershell
.\.venv\Scripts\python.exe .\30_start_detector_ui.py --threads 4
```

Open **http://127.0.0.1:8765** in your browser. Keep the terminal running; press Ctrl+C to stop it. If the port is busy, use `--port 8766` and open the printed address.

Choose **Website URL** for a public HTTP(S) source capture, or **Saved HTML** to paste a complete archived/rendered capture with its final page URL. The URL matters because form, script and link references are resolved against it. The rendered-capture option records what you supplied; it does not verify that a browser produced the HTML.

The result shows a static model score, capture method, and quality flags. `Potential phishing` means score ≥ 0.50. `No phishing predicted` is the model's output, not a guarantee of safety. A network failure, non-HTML response, HTTP error, incomplete/oversized response, undecodable HTML or empty page produces an inconclusive result without a model prediction. Recognized challenge/unavailable-page text also produces an inconclusive display; its raw model output remains diagnostic in the JSON.

Each completed request gets a new report under `reports/webpage_analysis_v1/`. You can also download that JSON from the interface. Page HTML is not embedded in the result or written to the report; reports contain the URL, capture metadata, hash, features, quality flags, and model score. URLs and redirects can contain sensitive query parameters, so review a report before sharing it.

## 4. Command-line use

Analyze complete saved HTML offline:

```powershell
.\.venv\Scripts\python.exe .\29_run_webpage_detector.py --html-file .\examples\example_page.html --page-url "https://example.com/login" --basis http_source
```

This example is a software demonstration, not a labelled accuracy test. Use `--basis browser_rendered` for a complete DOM capture you actually obtained through a browser.

Capture HTTP source and analyze it:

```powershell
.\.venv\Scripts\python.exe .\29_run_webpage_detector.py --url "https://example.com" --threads 4
```

The crawler follows up to five redirects, validates public IP addresses at each hop, pins each connection to the validated address, verifies TLS certificates, and bounds the response/decompressed body to 32 MiB. The nominal overall capture deadline is 45 seconds, with bounded socket waits; operating-system DNS resolution can add delay. It does not use session cookies, perform login, fetch external assets, execute JavaScript or bypass website challenges. IPv4 and IPv6 private/local/reserved targets are rejected. HTTP proxies from environment variables are not used.

The original feature analysis caps remain **4,000,000 HTML characters** and **500,000 inline-JS characters**. Those caps are separate from the complete-response byte limit. The extractor receives the full HTML length before its analysis cap and records truncation quality flags exactly as training did. External JavaScript references are counted; their code is not added to the static features.

## 5. Run the controlled dynamic lab

Install only the browser additions, preserving the existing ML packages:

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\requirements_runtime.txt
.\.venv\Scripts\python.exe -m playwright install chromium
.\.venv\Scripts\python.exe .\31_test_dynamic_runtime.py
```

The browser package version matches the original script-21 workflow. This test launches a fresh Chromium context for each trusted built-in fixture. All fixture-page requests are supplied from memory or aborted. WebSockets are blocked, service workers disabled, and only synthetic credential markers are entered.

Expected completion: **20/20 checks passed**. A new report is saved under `reports/dynamic_runtime_v1/`. You can show the browser with `--headed`.

| Test group | What it checks |
| --- | --- |
| 17 original behavior cases | Delayed forms/requests, redirects, fetch/XHR/beacon, direct form submission, associated credential labels, ordinary login/contact controls, a mock external script, encoded script, and unknown-request blocking |
| 3 rendered feature comparisons | Delayed, Base64-encoded and mock external JavaScript create a plaintext password-cued form; unchanged script 15 sees the added form in rendered DOM while raw source has no form |

The ordinary cross-origin login control intentionally also sends a synthetic password to another origin. This demonstrates why that observation alone must not be classified as phishing. Dynamic credential cues include broader languages/variants than the frozen static features and stay separate from classifier input. Cross-origin behavior compares scheme/host/port; the static feature extractor uses site groups.

The reused fixture implementation is bundled as `detector_runtime/controlled_fixtures.py`; it preserves the existing script-21 implementation without replacing your original file. Run the lab through script 31.

## Capture method and accuracy scope

PhreshPhish's collection used Selenium/Chrome to capture rendered dynamic page content. Your frozen test results therefore describe archived HTML under that collection process. A simple HTTP-source crawler can miss JavaScript-created HTML, even when it downloads the full HTTP response.

Sources: [PhreshPhish paper, collection methodology](https://arxiv.org/html/2507.10854v2), [Playwright Python installation](https://playwright.dev/python/docs/library), [browser installation](https://playwright.dev/python/docs/browsers), [request routing API](https://playwright.dev/python/docs/api/class-browsercontext), [WebSocket routing](https://playwright.dev/python/docs/api/class-websocketroute).

Your previously measured balanced, filtered static test result was **86.30% accuracy, 77.70% recall and 5.10% false-positive rate** at threshold 0.50. This runtime does not change those measurements. They are not measured accuracy for live URL capture or for a dynamic detector, and the filtered balanced cohort is not the complete released test population.

Passing the controlled lab checks establishes functioning instrumentation, not dynamic phishing accuracy, resistance to all evasion, or a safe sandbox for arbitrary hostile JavaScript. Do not use this fixture harness to execute unknown phishing pages on your laptop. Further dynamic classification needs independently labelled captures, development-only model/threshold decisions, benign controls such as SSO and dynamic contact forms, and a new independent holdout. The previously examined test cohort cannot serve as a fresh final holdout for improvements.

## Integration with a shared crawler/backend

From Python, initialize the detector once and pass complete HTML and its final URL:

```python
from detector_runtime.model import FrozenDetector

detector = FrozenDetector(r"D:\research_new\HTML_JS_PhishingDetector", threads=4)
result = detector.predict_html(
    complete_html,
    final_page_url,
    capture_basis="browser_rendered",  # only if your crawler actually renders
)
```

This method performs no network requests. The caller owns capture completeness, browser isolation and capture provenance. It returns a JSON-compatible dictionary. Dynamic observations should be reported alongside this result; do not append them to the 106-vector or silently use them to change the frozen classifier label. The bundled HTTP interface runs on loopback for local use, not as a public production service.

## Troubleshooting

- **Missing helper / changed hash:** keep the original script 09, script 15, schema, run config, model and freeze report. Do not regenerate or edit them to bypass the check; share the exact error.
- **DLL blocked by Application Control:** this is an OS policy/import problem. Keep Defender enabled. The runtime does not repair security policy. Share the blocked file/module message so the environment can be repaired through approved settings or package restoration.
- **Dependency mismatch:** use the original working `.venv`. Do not run a blanket upgrade of NumPy, scikit-learn, LightGBM or joblib.
- **Chromium missing:** run the browser-install command above. If it is still blocked by an OS policy, the static URL/HTML interface can run without Chromium; the dynamic lab remains unverified on that machine.
- **403/challenge/empty page:** obtain an authorized complete capture or report it as inconclusive. An inaccessible page is not a benign page.

## Verification supplied with this package

See `VERIFICATION.md`. The integration was tested with local HTTP fixtures, the exact original feature extractors, a separate temporary synthetic LightGBM bundle, and the trusted browser fixtures. The actual saved model bundle exists on your laptop and was not available for independent loading here; **step 2 is the check of your real artifact**.

Optional local regression tests after extracting into the existing project:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s .\tests -v
```

These tests train a small temporary synthetic fixture model to exercise the loader, feature order and inference plumbing. They do not train, change, select or evaluate your research model. Your normal runtime scripts never call model fitting.
