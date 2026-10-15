"""One browser attempt per labelled URL; transactional resumable checkpoints."""
import csv
import hashlib
import json
import random
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from live_evaluation.protocol import canonical_json, object_sha, classification_metrics
from .behavior import VERSION, analyze_browser_url


def utc():
    return datetime.now(timezone.utc).isoformat()


def summarize(rows,records,complete):
    scored=[(row['label'],records[row['ordinal']]['analysis']['evaluated_prediction'])
        for row in rows if row['ordinal'] in records
        and records[row['ordinal']]['analysis'].get('status')=='prediction_complete'
        and records[row['ordinal']]['analysis'].get('evaluated_prediction') in {'benign','phish'}]
    selected=Counter(row['label'] for row in rows)
    scored_counts=Counter(label for label,pred in scored)
    outcomes=Counter(record['analysis']['status'] for record in records.values())
    paired=[(row['label'],records[row['ordinal']]['analysis']) for row in rows
        if row['ordinal'] in records
        and records[row['ordinal']]['analysis'].get('status')=='prediction_complete'
        and records[row['ordinal']]['analysis'].get('evaluated_prediction') in {'benign','phish'}
        and records[row['ordinal']]['analysis'].get('source_analysis',{}).get('status')=='prediction_complete']
    return {'status':'browser_evaluation_complete' if complete else 'browser_evaluation_partial',
        'selected':len(rows),'attempted':len(records),'scored':len(scored),
        'pending':len(rows)-len(records),'outcomes':dict(outcomes),
        'selected_class_counts':dict(selected),'scored_class_counts':dict(scored_counts),
        'scored_coverage':len(scored)/len(rows),
        'coverage_per_class':{label:scored_counts[label]/selected[label] for label in selected},
        'classification_metrics_on_scored_pages':classification_metrics(scored),
        'paired_source_rendered_comparison':{
            'paired_pages':len(paired),
            'scope':'Same browser attempts with both source and rendered predictions complete; this is a conditional subset.',
            'source_metrics':classification_metrics([(label,a['source_analysis']['static_model_prediction']) for label,a in paired]),
            'rendered_metrics':classification_metrics([(label,a['evaluated_prediction']) for label,a in paired])},
        'warnings':['Metrics exclude failures and inconclusive captures; report coverage with accuracy.',
          'This browser pipeline is a different capture method from step 32; do not replace the old measurements.',
          'Labels are supplied independently. Feed labels may be stale when a page is captured.',
          'URLs examined during development are not a fresh final holdout.']}


def atomic_json(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('w',encoding='utf-8') as handle:
        json.dump(value,handle,indent=2,ensure_ascii=True,allow_nan=False)
        handle.write('\n')
    temp.replace(path)


def run_evaluation(detector,rows,raw,output,code_hashes,analyzer=analyze_browser_url,progress=None):
    output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    identity={'pipeline':VERSION,'input_sha256':hashlib.sha256(raw).hexdigest(),
              'rows':rows,'model':detector.describe(),'code_sha256':code_hashes,
              'observation_ms':6000,'interaction_ms':1500,'shuffle_seed':42}
    conn=sqlite3.connect(output/'checkpoint.sqlite3',timeout=0)
    # One runner holds a separate SQLite lock for its whole lifetime.
    lock=sqlite3.connect(output/'run_lock.sqlite3',timeout=0)
    try:
        lock.execute('CREATE TABLE IF NOT EXISTS runner (n INTEGER)');lock.commit()
        lock.execute('BEGIN EXCLUSIVE')
        conn.execute('CREATE TABLE IF NOT EXISTS protocol (identity TEXT NOT NULL)')
        conn.execute('CREATE TABLE IF NOT EXISTS attempts (ordinal INTEGER PRIMARY KEY, state TEXT NOT NULL, payload TEXT)')
        stored=conn.execute('SELECT identity FROM protocol').fetchall()
        if stored:
            if len(stored)!=1 or stored[0][0]!=canonical_json(identity):
                raise ValueError('Input, model, settings or code changed. Keep the original run intact and choose a new output folder.')
            if (output/'frozen_input.csv').read_bytes()!=raw:
                raise ValueError('Frozen input changed.')
            saved_protocol=json.loads((output/'evaluation_protocol.json').read_text(encoding='utf-8'))
            if saved_protocol.get('identity')!=identity or saved_protocol.get('identity_sha256')!=object_sha(identity):
                raise ValueError('Saved evaluation protocol changed.')
        else:
            if any(p.name not in {'checkpoint.sqlite3','run_lock.sqlite3'} for p in output.iterdir()):
                raise ValueError('Choose a new empty report folder.')
            (output/'frozen_input.csv').write_bytes(raw)
            conn.execute('INSERT INTO protocol VALUES (?)',(canonical_json(identity),));conn.commit()
            atomic_json(output/'evaluation_protocol.json',{'identity':identity,'identity_sha256':object_sha(identity),'created_utc':utc()})
        records={}
        for ordinal,state,payload in conn.execute('SELECT ordinal,state,payload FROM attempts'):
            if ordinal not in {row['ordinal'] for row in rows}:
                raise ValueError('Checkpoint contains an unexpected row.')
            if state=='in_progress':
                record={'ordinal':ordinal,'analysis':{'status':'interrupted_capture','evaluated_prediction':None,
                    'error':'The previous attempt was interrupted. It is retained without another website request.'}}
                record['record_sha256']=object_sha(record)
                conn.execute('UPDATE attempts SET state=?,payload=? WHERE ordinal=?',('done',canonical_json(record),ordinal))
            elif state=='done':
                record=json.loads(payload)
                if record['ordinal']!=ordinal or not isinstance(record.get('analysis'),dict):
                    raise ValueError('Invalid saved attempt.')
                checksum=record.pop('record_sha256',None)
                if checksum!=object_sha(record):
                    raise ValueError('Saved attempt failed its integrity check.')
                record['record_sha256']=checksum
            else:
                raise ValueError('Unknown checkpoint state.')
            records[ordinal]=record
        conn.commit()
        order=list(rows);random.Random(42).shuffle(order)
        for row in order:
            ordinal=row['ordinal']
            if ordinal in records:
                continue
            conn.execute('INSERT INTO attempts VALUES (?,?,NULL)',(ordinal,'in_progress'));conn.commit()
            start=utc()
            try:
                analysis=analyzer(detector,row['normalized_url'])
            except Exception as error:
                analysis={'status':'analysis_failed','evaluated_prediction':None,'error':f'{type(error).__name__}: {error}'}
            record={'ordinal':ordinal,'started_utc':start,'ended_utc':utc(),'analysis':analysis}
            record['record_sha256']=object_sha(record)
            conn.execute('UPDATE attempts SET state=?,payload=? WHERE ordinal=?',('done',canonical_json(record),ordinal));conn.commit()
            records[ordinal]=record
            if progress:
                progress(len(records),len(rows),row['csv_row'],analysis['status'])
        complete=len(records)==len(rows)
        report=summarize(rows,records,complete)
        report['protocol_sha256']=object_sha(identity)
        atomic_json(output/'browser_evaluation_report.json',report)
        with (output/'browser_evaluation_predictions.csv').open('w',encoding='utf-8-sig',newline='') as handle:
            names=['csv_row','url','label','label_source','label_checked_at','status','evaluated_prediction','source_score','rendered_score','error']
            writer=csv.DictWriter(handle,fieldnames=names);writer.writeheader()
            for row in rows:
                a=records[row['ordinal']]['analysis']
                value={k:row[k] for k in names[:5]}
                value.update(status=a['status'],evaluated_prediction=a.get('evaluated_prediction'),
                    source_score=a.get('source_analysis',{}).get('phishing_score'),rendered_score=a.get('phishing_score'),error=a.get('error',''))
                writer.writerow({k:("'"+v if isinstance(v,str) and v.lstrip().startswith(('=','+','-','@')) else v) for k,v in value.items()})
        return report
    finally:
        conn.close()
        lock.close()
