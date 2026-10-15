# Step 33 — review the completed live evaluation

This step reads the results and captured page source already saved by script 32. It helps you understand the six incorrect predictions and the capture-quality limitations before developing further changes.

The existing run remains the preliminary live pilot: 100 attempted URLs, 58 scored pages, 42 capture failures, and 89.66% accuracy on the scored pages. This review does not create a new accuracy result.

## 1. Copy the add-on into your project

Extract the ZIP and copy these files into:

```text
D:\research_new\HTML_JS_PhishingDetector
```

The resulting layout should include:

```text
HTML_JS_PhishingDetector/
  33_review_live_errors.py
  STEP_33_READ_ME.md
  tests/
    test_live_error_review.py
  reports/
    live_evaluation_local_20261011/
      live_evaluation_report.json
      live_evaluation_predictions.csv
      frozen_input.csv
      evaluation_protocol.json
      checkpoint.sqlite3
```

Copy the new test file into your existing `tests` folder. Keep the existing test files there. The add-on contains no replacement model, dataset, runtime or scripts 01–32.

## 2. Check the saved run

Open the VS Code PowerShell terminal at the project folder and run:

```powershell
Set-Location "D:\research_new\HTML_JS_PhishingDetector"
.\.venv\Scripts\python.exe .\33_review_live_errors.py --check-only
```

The check verifies that the frozen input, protocol, checkpoint, prediction CSV and report agree. It does not visit any URLs. If it reports a mismatch, preserve the original files and share the exact error. The checks detect inconsistent or damaged artifacts; they do not independently prove the original labels.

## 3. Generate the review

```powershell
.\.venv\Scripts\python.exe .\33_review_live_errors.py
```

Wait for:

```text
Live error review complete.
```

Reports are saved separately under:

```text
reports\live_error_review_v1
```

The original evaluation folder is read as evidence. No website capture, external-script download, JavaScript execution, model loading, training or threshold change is performed.

## 4. Read the review

- Open `live_error_review_summary.md` for the counts, incorrect predictions, capture failures and source-inspection notes.
- Open `review_index.csv` in Excel for the row-by-row review of the selected URLs.
- Preserve `live_error_review_report.json` as the detailed review evidence.
- Any saved source examples end in `.html.txt`; read them as text in VS Code. They are bounded inspection copies, not rendered pages. Their completeness and character limits are recorded in the review.

The review identifies observations such as interstitials, low visible text with scripts, declared navigation, JavaScript-dependent shells and duplicate captures. A flag is a reason to inspect the capture, not proof of phishing, an incorrect label or an invalid page. External scripts and pages without forms are common on legitimate websites too.

Source inspection uses bounded scans: the first 2,097,152 markup characters, at most 512 nested elements, 100,000 characters of source text and 1 MiB of inline script text. Cut flags disclose incomplete scans. Up to 24 source copies of at most 2 MiB each are saved, prioritizing incorrect predictions. Any omitted or truncated copies are recorded; every original row remains in the review.

The six classification errors, original labels and original scores stay in the review. The 42 failed captures stay separate. No rows are removed to improve the accuracy. The review does not calculate an alternative accuracy after excluding the flagged examples.

## 5. What follows this review

Use the observations to define capture-quality reporting rules before a future evaluation. If these results guide a detector or capture-rule change, treat them as development evidence and evaluate the revised system on a fresh independently labelled set.

The static live HTTP-source pilot and the separate controlled JavaScript lab answer different questions. This review does not measure dynamic phishing detection accuracy.

## Other saved runs

To review another complete script-32 output folder, name it explicitly:

```powershell
.\.venv\Scripts\python.exe .\33_review_live_errors.py --run-dir ".\reports\OTHER_COMPLETED_RUN" --output-dir ".\reports\ANOTHER_REVIEW_FOLDER"
```

Use the actual complete run's folder name. Keep reviews in separate folders. The original capture times identify the run; this is an offline inspection, not a second live test.

If the review output folder already contains files, the script asks you to choose a new `--output-dir`. Keep the existing review and name a new folder such as `reports\live_error_review_v2`.

## Optional focused software tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s .\tests -p test_live_error_review.py -v
```

These tests use synthetic artifacts to check artifact consistency and inspection behavior. They do not train or evaluate your research model.
