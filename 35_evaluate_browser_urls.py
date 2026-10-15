"""Evaluate the integrated browser detector on an independently labelled list."""
import argparse
from pathlib import Path
from live_evaluation.protocol import load_url_list,file_sha
from detector_runtime.capture import normalize_url
from detector_runtime.model import FrozenDetector
from detector_runtime.browser_evaluation import run_evaluation


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--urls',type=Path,required=True)
    parser.add_argument('--project-dir',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--per-class',type=int,default=50)
    parser.add_argument('--check-list',action='store_true')
    args=parser.parse_args()
    try:
        rows,raw=load_url_list(args.urls,normalize_url,args.per_class)
        print(f'Validated {args.per_class} benign + {args.per_class} phish URLs.')
        if args.check_list:
            print('Offline list check passed. No browser launched or report written.')
            return 0
        detector=FrozenDetector(args.project_dir)
        base=Path(__file__).resolve().parent
        paths=[Path(__file__).resolve(),*sorted((base/'detector_runtime').glob('*.py')),
               base/'detector_runtime'/'behavior_probe.js',base/'live_evaluation'/'protocol.py']
        hashes={str(path.relative_to(base)):file_sha(path) for path in paths}
        report=run_evaluation(detector,rows,raw,args.output_dir or args.project_dir/'reports'/'browser_url_evaluation_v1',hashes,
            progress=lambda done,total,row,status:print(f'Completed {done}/{total}; CSV row {row}: {status}',flush=True))
        metrics=report['classification_metrics_on_scored_pages']
        print('Browser evaluation complete.')
        print(f"Scored: {report['scored']}/{report['selected']}; coverage: {report['scored_coverage']:.2%}")
        if metrics['accuracy'] is not None:
            print(f"Accuracy on scored pages: {metrics['accuracy']:.2%}")
        else:
            print('Accuracy unavailable: no pages could be scored.')
        return 0
    except KeyboardInterrupt:
        print('Interrupted. Rerun the same command to resume; the interrupted attempt stays recorded.')
        return 130
    except Exception as error:
        print(f'Browser evaluation stopped: {type(error).__name__}: {error}')
        return 1


if __name__=='__main__':
    raise SystemExit(main())
