from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[4]
def resolve(path):return ROOT/str(path).replace('\\','/')
def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,ensure_ascii=False));temp.replace(path)
def verify(rows):
    for r in rows:
        assert hashlib.sha256(resolve(r['path']).read_bytes()).hexdigest()==r['sha256'],r['path']
def audit_dependencies(rows):
    out=[]
    for r in rows:
        b=resolve(r['path']).read_bytes();actual=hashlib.sha256(b).hexdigest()
        crlf=hashlib.sha256(b.replace(b'\r\n',b'\n').replace(b'\n',b'\r\n')).hexdigest()
        status='exact' if actual==r['sha256'] else ('line endings only' if crlf==r['sha256'] else 'source mismatch; numerical parity required')
        assert status!='source mismatch; numerical parity required' or r['path']=='code/core/mfdfa.py'
        out.append(dict(r,current_sha256=actual,status=status))
    atomic_json(ROOT/'result/test/step6/config/dependency_audit.json',out)
