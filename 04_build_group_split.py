"""Create a provisional, domain-separated training/validation manifest.

Uses only the training metadata produced by script 02. Standard library only.
No dataset files are modified/copied and no test records are read. All usable
rows keep their supplied labels. Rows without a site group are quarantined in
the output manifest. HTML similarity checks are still required before training.
"""
import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

LABELS = ("benign", "phish")


def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def choose_validation(groups, fraction, seed, trials=128, swap_attempts=100000):
    totals = [sum(counts[i] for counts in groups.values()) for i in range(2)]
    targets = [total * fraction for total in totals]
    if not all(totals):
        raise ValueError("Usable training metadata must contain both labels.")
    order = sorted(groups)
    if len(order) < 2:
        raise ValueError("At least two usable website groups are required.")
    validation_size = min(len(order) - 1, max(1, round(len(order) * fraction)))
    def score(counts):
        return sum(((counts[i] - targets[i]) / targets[i]) ** 2 for i in range(2))
    # Fix the validation GROUP count before balancing page counts. Version 1.0
    # optimized page counts alone and could put only a few huge sites in validation.
    # Choose the best class-balanced candidate among seeded whole-group samples;
    # neither page contents nor model results influence this choice.
    rng = random.Random(seed)
    best = None
    for trial in range(trials):
        candidate = rng.sample(order, validation_size)
        counts = [sum(groups[group][i] for group in candidate) for i in range(2)]
        if not all(0 < counts[i] < totals[i] for i in range(2)):
            continue
        candidate_score = score(counts)
        if best is None or candidate_score < best[0]:
            best = (candidate_score, trial, candidate, counts)
    if best is None:
        raise ValueError("No viable group split found. More --trials may help, but both classes must occur in multiple groups.")
    _, best_trial, validation_list, current = best
    selected = set(validation_list)
    training_list = [group for group in order if group not in selected]
    swaps_accepted = 0
    # One group in / one group out preserves broad group coverage throughout.
    for _ in range(swap_attempts):
        vi = rng.randrange(len(validation_list))
        ti = rng.randrange(len(training_list))
        out_group, in_group = validation_list[vi], training_list[ti]
        proposed = [current[i] - groups[out_group][i] + groups[in_group][i] for i in range(2)]
        if not all(0 < proposed[i] < totals[i] for i in range(2)):
            continue
        if score(proposed) < score(current) - 1e-15:
            selected.remove(out_group)
            selected.add(in_group)
            validation_list[vi], training_list[ti] = in_group, out_group
            current = proposed
            swaps_accepted += 1
    if len(selected) != validation_size:
        raise ValueError("Internal group-count consistency check failed.")
    return selected, totals, current, {
        "candidate_trials": trials, "best_initial_trial_zero_based": best_trial,
        "swap_attempts": swap_attempts, "swaps_accepted": swaps_accepted,
    }


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=root / "reports" / "domains" / "training_domain_manifest.csv")
    parser.add_argument("--domain-report", type=Path,
                        default=root / "reports" / "domains" / "domain_audit_report.json")
    parser.add_argument("--output-dir", type=Path,
                        default=root / "reports" / "split")
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trials", type=int, default=128)
    args = parser.parse_args()
    if not 0.05 <= args.validation_fraction <= 0.50:
        parser.error("--validation-fraction must be between 0.05 and 0.50")
    if not 1 <= args.trials <= 1024:
        parser.error("--trials must be between 1 and 1024")
    for path in (args.manifest, args.domain_report):
        if not path.is_file():
            parser.error(f"Missing input: {path}. Run script 02 first.")
    audit = json.loads(args.domain_report.read_text(encoding="utf-8-sig"))
    if audit["training_missing_hash_rows"] or audit["training_repeated_supplied_hash_rows"]:
        parser.error("This script requires non-missing, unique supplied training hashes.")
    print("Reading training domain manifest...", flush=True)
    required = {"sha256", "label", "date", "source_file", "hostname", "site_group", "domain_status"}
    groups = defaultdict(lambda: [0, 0])
    labels = Counter()
    statuses = Counter()
    hashes = set()
    quarantined = []
    with args.manifest.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        fields = reader.fieldnames
        if not fields or not required.issubset(fields):
            parser.error("Input manifest does not have script 02's required columns.")
        if "split" in fields or "split_reason" in fields:
            parser.error("Input must be the original training domain manifest, not an existing split.")
        for index, row in enumerate(reader, 1):
            label, sha, group = row["label"], row["sha256"], row["site_group"]
            if label not in LABELS:
                parser.error(f"Unexpected label on row {index}: {label}")
            if not sha or sha in hashes:
                parser.error(f"Missing or duplicate supplied hash on row {index}.")
            hashes.add(sha)
            labels[label] += 1
            statuses[row["domain_status"]] += 1
            if not group:
                quarantined.append({key: row[key] for key in required})
                continue
            if not row["hostname"]:
                parser.error(f"Site group without a hostname on row {index}.")
            groups[group][LABELS.index(label)] += 1
    if dict(labels) != audit["training_label_counts"]:
        parser.error("Manifest label counts differ from the domain report.")
    if dict(statuses) != audit["training_domain_status_counts"]:
        parser.error("Manifest domain status counts differ from the domain report.")
    if len(groups) != audit["training_unique_site_groups"]:
        parser.error("Manifest group count differs from the domain report.")
    del hashes
    print(f"Assigning {len(groups):,} intact website groups...", flush=True)
    try:
        validation_groups, totals, val_counts, selection_details = choose_validation(
            groups, args.validation_fraction, args.seed, trials=args.trials)
    except ValueError as error:
        parser.error(str(error))
    training_groups = set(groups) - validation_groups
    assert not (training_groups & validation_groups)
    manifest_digest = file_digest(args.manifest)
    audit_digest = file_digest(args.domain_report)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "training_validation_manifest.csv"
    if output.resolve() in (args.manifest.resolve(), args.domain_report.resolve()):
        parser.error("Output must not overwrite an input file.")
    temporary = output.with_suffix(".csv.tmp")
    counts = {name: Counter() for name in ("train", "validation", "quarantine")}
    with args.manifest.open(newline="", encoding="utf-8-sig") as source, \
            temporary.open("w", newline="", encoding="utf-8") as destination:
        reader = csv.DictReader(source)
        writer = csv.DictWriter(destination, fieldnames=fields + ["split", "split_reason"])
        writer.writeheader()
        for row in reader:
            group = row["site_group"]
            split = "quarantine" if not group else (
                "validation" if group in validation_groups else "train")
            reason = "unresolved_url_group" if not group else "intact_site_group"
            counts[split][row["label"]] += 1
            writer.writerow(dict(row, split=split, split_reason=reason))
    if file_digest(args.manifest) != manifest_digest:
        temporary.unlink(missing_ok=True)
        raise SystemExit("Input changed during the run. No completed split written.")
    assert sum(sum(count.values()) for count in counts.values()) == sum(labels.values())
    assert [counts["validation"][label] for label in LABELS] == val_counts
    temporary.replace(output)
    def largest(group_set):
        ordered = sorted(group_set, key=lambda group: (-sum(groups[group]), group))[:10]
        return [{"site_group": group, "rows": sum(groups[group]),
                 "benign": groups[group][0], "phish": groups[group][1]} for group in ordered]
    fraction_actual = sum(val_counts) / sum(totals)
    fractions_label = {label: val_counts[i] / totals[i] for i, label in enumerate(LABELS)}
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "script_version": "1.1",
        "status": "provisional_until_html_similarity_check", "seed": args.seed,
        "method": "Fixed validation group count; 128 seeded whole-group candidate samples by default, selected for class balance; whole-group swaps improve class balance while preserving group count.",
        "selection_details": selection_details,
        "validation_fraction_requested": args.validation_fraction,
        "validation_fraction_actual": fraction_actual,
        "validation_fraction_by_label": fractions_label,
        "input_manifest_sha256": manifest_digest, "input_domain_report_sha256": audit_digest,
        "input_training_rows": sum(labels.values()), "usable_rows": sum(totals),
        "label_counts_by_split": {name: {label: counts[name][label] for label in LABELS}
                                  for name in counts},
        "group_counts_by_split": {"train": len(training_groups), "validation": len(validation_groups)},
        "validation_group_fraction_actual": len(validation_groups) / len(groups),
        "shared_train_validation_site_groups": len(training_groups & validation_groups),
        "supplied_training_hashes_unique": True,
        "quarantined_records": quarantined,
        "largest_train_groups": largest(training_groups),
        "largest_validation_groups": largest(validation_groups),
        "largest_validation_group_row_fraction": max(sum(groups[group]) for group in validation_groups) / sum(val_counts),
        "top_ten_validation_groups_row_fraction": sum(item["rows"] for item in largest(validation_groups)) / sum(val_counts),
        "class_balance_within_one_percentage_point": all(abs(value - args.validation_fraction) <= 0.01
                                                         for value in fractions_label.values()),
        "limitations": [
            "This is a metadata-based split; no HTML similarity check has been completed here.",
            "Site groups follow script 02 including conservative platform-level groups where the bundled suffix list does not separate tenants.",
            "Unknown-suffix exact-hostname groups are retained; rows without a group are quarantined only in this manifest.",
            "Labels are retained as supplied, including mixed-label website groups; label accuracy is not independently verified.",
            "Exact validation percentages are not guaranteed because groups remain indivisible.",
            "Validation group count targets the requested fraction independently of page counts; large groups can still dominate some metrics.",
            "Pages on different domains can share templates or campaigns. A domain-separated split alone cannot eliminate all leakage.",
            "Official test data are not read, altered, or used to select the validation groups.",
            "No features are extracted and no model is trained by this script.",
        ],
    }
    report_path = args.output_dir / "split_report.json"
    report_temporary = report_path.with_suffix(".json.tmp")
    report_temporary.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    report_temporary.replace(report_path)
    print(json.dumps(report["label_counts_by_split"], indent=2), flush=True)
    print(f"Validation fraction: {fraction_actual:.2%}", flush=True)
    print(f"Validation website groups: {len(validation_groups):,}/{len(groups):,} ({len(validation_groups)/len(groups):.2%})", flush=True)
    print("Shared train/validation website groups: 0", flush=True)
    if not report["class_balance_within_one_percentage_point"]:
        print("Class balance differs from the target by more than one percentage point; review the report.", flush=True)
    print(f"Saved report: {report_path.resolve()}", flush=True)
    print(f"Saved manifest: {output.resolve()}", flush=True)
    print("Group split complete. HTML similarity check still required before training.", flush=True)


if __name__ == "__main__":
    main()
