# Balanced live URL evaluation — step 32

This add-on measures your **existing frozen static detector** on a new, labelled URL list. It does not train a model. The default pilot is **50 legitimate (`benign`) + 50 phishing (`phish`) URLs**. The model stays `lightgbm_context_all_features`, with 106 features, best iteration 1269 and threshold 0.50.

## 1. Extract into your existing project

Extract the ZIP contents directly into:

```text
D:\research_new\HTML_JS_PhishingDetector
```

You should then have `32_evaluate_live_urls.py`, `live_evaluation`, and `inputs` beside scripts 29–31 and `detector_runtime`. Keep your original scripts, frozen reports, model files and `.venv`. This add-on supplies no replacement model or old runtime files. No additional package installation is needed.

## 2. Fill the URL list before running predictions

Open `inputs\live_test_urls_template.xlsx` in Excel. It has 100 slots: 50 `benign` and 50 `phish`. Fill the other three fields for every row. Then **Save As → CSV UTF-8 (Comma delimited)**, using the filename:

```text
inputs\live_test_urls.csv
```

You may replace the blank CSV included in this ZIP when saving your filled list. Fill either the workbook or CSV; they are two formats of the same template. The evaluator reads the CSV, not the XLSX. Keep the four column names exactly as supplied.

| Field | What you enter |
| --- | --- |
| `url` | Complete public `https://...` or `http://...` page URL. |
| `label` | `benign` for independently verified legitimate pages; `phish` for independently verified phishing pages. The template presets the required counts. |
| `label_source` | Evidence for the label: for phishing, preferably the corresponding PhishTank detail-page URL; for legitimate sites, an official ownership/reference URL or a clear independent verification note. |
| `label_checked_at` | The date **you checked that evidence**, in `YYYY-MM-DD` format. Do not invent or copy a verification date you did not check. |

Use fresh URLs that you have not already selected based on this detector's predictions. The links used for your earlier demonstration should remain demonstration links. Avoid several URLs from the same website; aim for different sites in each class. Include legitimate login/forms and varied ordinary pages, rather than only simple homepages. Record how you selected the sites.

For PhishTank, choose entries marked verified and online **at collection time**. Preserve the detail URL and check date. A feed label does not guarantee the page stays online or unchanged when you later fetch it. See the official [PhishTank feed documentation](https://phishtank.net/developer_info.php).

Do not use the model's prediction as evidence for the label, choose only links it gets right, or change a label because it disagrees. The evaluator checks completeness and balance; it **cannot verify ground truth**. Overlap with PhreshPhish training domains is not automatically assessed, so describe this as a live pilot rather than a proven unseen-domain benchmark.

## 3. Check the list without visiting websites

In the VS Code PowerShell terminal, at the project folder:

```powershell
.\.venv\Scripts\python.exe .\32_evaluate_live_urls.py --urls ".\inputs\live_test_urls.csv" --check-list
```

Fix any listed row errors. The check requires exactly 50 URLs per class, unique normalized URLs, evidence and valid dates. It does not load the classifier, fetch websites or write reports. If Excel saved semicolons instead of commas, export comma-delimited CSV UTF-8 again.

## 4. Run the evaluation

```powershell
.\.venv\Scripts\python.exe .\32_evaluate_live_urls.py --urls ".\inputs\live_test_urls.csv" --threads 4
```

The program verifies your saved model/runtime and freezes the list, labels, versions and capture settings before predictions. It requests pages in a reproducibly shuffled order (seed 42). It captures HTTP source as text, with the existing 32 MiB size limit, five-redirect limit and 45-second capture timeout. It does not run page JavaScript, download external scripts, submit forms or supply credentials.

Keep the computer awake while it runs. If a request fails or is blocked, leave it recorded; keep Windows Defender enabled. Do not change labels or replace failed pages in the same evaluation. A recognized challenge/unavailable-page response is recorded as an inconclusive capture. Other capture limitations can still exist and are reflected in quality flags.

After an interruption or inference error, rerun **the same command** using the same CSV, runtime and output folder. Completed rows are skipped. Saved HTML from an interrupted inference is reused without another request. A capture interrupted mid-request is retained as `interrupted_capture` rather than silently retried. Each selected URL has one recorded attempt; no automatic replacement or repeated attempts to obtain a favorable result.

## 5. Read and share the results

Wait for the final message **`Live evaluation complete.`** A partial report is not the final result. Reports are saved in:

```text
reports\live_evaluation_v1\live_evaluation_report.json
reports\live_evaluation_v1\live_evaluation_predictions.csv
```

Upload those two files for review. The JSON contains the confusion matrix, accuracy, precision, recall, F1, false-positive rate and coverage per class. The CSV contains every selected URL, its original label, outcome, error and quality flags. Failed/inconclusive rows have no evaluated prediction. Scores for inconclusive pages appear only in the diagnostic columns.

**Report coverage with accuracy.** The selected list is balanced, but capture failures may make the scored subset unbalanced. The program reports that difference and retains all successful pages. Classification metrics use only `prediction_complete` pages; capture failures are not counted as correct benign predictions. The extra fraction of correct scored predictions over all selected URLs is clearly labelled as a coverage-sensitive fraction, not classification accuracy. Undefined rates are JSON `null`.

False-positive rate = legitimate pages incorrectly predicted phishing / scored legitimate pages. Recall = phishing pages correctly detected / scored phishing pages. A balanced 50+50 pilot is an initial evaluation, not strong evidence of deployment accuracy. For example, zero false positives among 50 scored legitimate pages still has an approximate Wilson 95% interval reaching 7.13%. Correlated sites and nonrandom selection add uncertainty beyond that interval.

The model was developed with archived browser-rendered HTML. This pilot measures live **HTTP-source static detection**. It does not measure JavaScript execution, delayed phishing detection or the separate controlled dynamic-runtime component. Report it separately from the earlier filtered PhreshPhish test results. Keep the pilot results out of model tuning if you want to preserve their evaluation role; a later improved model needs a new evaluation set.

Preserve the whole output folder for reproducibility. `frozen_input.csv` stores the exact supplied labels. `evaluation_protocol.json` stores the model/code/input fingerprints and request order. `checkpoint.sqlite3` stores captured HTML text, capture metadata and full model results transactionally. HTML is stored as compressed text in the checkpoint, not executed. Duplicate final URLs or HTML are flagged for review; no rows are automatically dropped or relabelled.

## Optional commands

A smaller or larger **separately planned** balanced pilot may use `--per-class N`. Keep that value in the command for both checking and running; the CSV must then contain exactly N rows of each class. The supplied template and default command use 50.

For a deliberately new run with its own selection/time window, choose a new output directory and describe it as a separate evaluation:

```powershell
.\.venv\Scripts\python.exe .\32_evaluate_live_urls.py --urls ".\inputs\live_test_urls.csv" --threads 4 --output-dir ".\reports\live_evaluation_v2"
```

Do not delete checkpoints to restart the same evaluation or report only the best run.

The optional offline tests require the original runtime folder but no fitted model or new libraries:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s .\tests -p test_live_url_evaluation.py -v
```

Tests use predetermined stub scores and a controlled local HTTP fixture. Passing them verifies the evaluation plumbing, not phishing accuracy. Your actual saved model is verified when the real run starts.
