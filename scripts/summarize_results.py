"""Extract the same public metrics from a regenerated architecture run."""
import argparse
import csv
import json
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',default='runs/architecture')
    p.add_argument('--output',default='runs/architecture/dev_metrics.csv')
    a=p.parse_args()
    root=Path(a.root)
    plan=json.loads((root/'protocol.json').read_text())['plan']
    rows=[]
    metrics={'retrieval':'ndcg@10','rerank':'ndcg@10','classification':'accuracy','ner':'entity_micro_f1'}
    for name in plan['names']:
        row={'configuration':name}
        for task,key in metrics.items():
            for lang in ('en','zh'):
                result=json.loads((root/f'evaluations/dev/{name}_42/{task}_{lang}.json').read_text())
                row[f'{task}_{lang}']=result['metrics'][key]
        rows.append(row)
    keys=list(rows[0])[1:]
    for row in rows:
        row['mean_gain']=sum(row[k]-rows[0][k] for k in keys)/8
    dest=Path(a.output)
    dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    print(dest)

if __name__=='__main__':
    main()
