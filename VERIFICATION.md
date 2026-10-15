# Integration verification — 9 October 2026

**16 regression tests passed. 20 controlled browser checks passed. Desktop and mobile interface checks passed.**

These results validate this package's software integration. They are not new phishing accuracy results.

| Area | Evidence |
| --- | --- |
| Original extractor contract | Byte-verified scripts 09 and 15; feature names/order, definitions, caps, association limits and all 11 quality fields match the frozen 106-feature schema |
| Inference parity | Float32 vectors and scores match direct LightGBM booster prediction at the saved best iteration using a temporary synthetic bundle |
| Loader integrity | Changed model bytes rejected before joblib deserialization; changed feature order/threshold rejected; completed offline CLI check writes a separate report |
| Source capture | Local HTTP fixtures cover complete source/gzip captures and redirect provenance; incomplete, non-HTML, forbidden, undecodable and oversized responses fail without a model label |
| Public address validation | Private/mixed DNS answers and redirects to private addresses rejected; each real URL hop pins its connection to the validated address |
| UI/backend | Saved-HTML submission reaches inference and writes JSON; cross-origin submission rejected; private URL produces an inconclusive result without a displayed score |
| Browser behavior | All 17 original fixture cases pass, including delayed/encoded/external-script scenarios and ordinary login/contact controls |
| Source/rendered gap | Three additional fixtures expose a dynamically added credential-cued form to the unchanged feature extractor; raw source contains no form |
| Visual checks | Desktop result/inconclusive screens and a 390-pixel mobile screen inspected; no horizontal overflow or page JavaScript errors |

Test environment: Linux, Python 3.12.14, NumPy 2.5.3, scikit-learn 1.9.1, LightGBM 4.7.0, joblib 1.6.0, threadpoolctl 3.7.0, DuckDB 1.5.6, tldextract 5.4.0. Controlled browser tests used Playwright 1.55.0 and Chromium 140.0.7339.16. The browser lab completed in approximately 30 seconds here; laptop timing can differ.

The temporary model used only synthetic numerical data for plumbing tests and was created in a disposable directory. It is not included in the package and never replaced the user's research model. The production runtime has no bypass option for the original frozen specification or artifact hashes.

The user's actual model bundle was not supplied for this integration task. Run `29_run_webpage_detector.py --check` in the original project `.venv` to verify that real artifact on Windows. Native Windows DLL/security-policy behavior has not been independently tested here.

The full controlled browser observations are included at `verification/controlled_dynamic_report.json`. Fixture roles/expected checks are software-test metadata, not phishing ground-truth labels. No live phishing pages were visited, and no dynamic phishing accuracy was measured.
