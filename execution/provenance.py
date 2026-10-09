"""Portable paths and strict, non-rebaselining provenance migration."""
from pathlib import Path, PurePosixPath
import csv, hashlib, json, os
ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'result/test/colab_migration'
LOCK = ROOT / 'result/test/step5/config/locked_step4_configuration.json'
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()
def resolve(value, root=ROOT):
    p=PurePosixPath(str(value).replace('\\','/'))
    if p.is_absolute() or '..' in p.parts or ':' in str(p): raise ValueError('Expected project-relative path: '+str(value))
    return Path(root).joinpath(*p.parts)
def normalized(value):
    if isinstance(value,list): return [normalized(x) for x in value]
    if isinstance(value,dict):
        return {k:(v.replace('\\','/') if k in ('path','relative_source','array_path','diagnostics_path') and isinstance(v,str) else normalized(v)) for k,v in value.items()}
    return value
def config_hash(config):
    return hashlib.sha256(json.dumps({k:v for k,v in config.items() if k!='config_sha256'},sort_keys=True).encode()).hexdigest()
def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8');os.replace(temp,path)
def verify(entries):
    errors=[]
    for item in entries:
        p=resolve(item['path']); actual=sha(p) if p.is_file() else 'MISSING'
        if actual!=item['sha256']: errors.append(dict(path=item['path'],expected=item['sha256'],actual=actual))
    if errors: raise RuntimeError('Dependency MISSING/CHANGED; refusing to rebaseline:\n'+json.dumps(errors,indent=2))
    return len(entries)
def references():
    base=ROOT/'result/test/step5/config'; entries=[]
    for name in ('locked_step4_configuration.json','runtime_provenance.json'):
        entries+=json.loads((base/name).read_text(encoding='utf-8'))['dependencies']
    with (base/'reused_step4_array_manifest.csv').open(encoding='utf-8-sig',newline='') as f: entries+=list(csv.DictReader(f))
    entries += [dict(path=x['relative_source'],sha256=x['sha256']) for x in json.loads((base/'source_inventory.json').read_text(encoding='utf-8'))]
    return entries
def migrate():
    checked=verify(references()) # ALL dependencies BEFORE any mutation.
    old=json.loads(LOCK.read_text(encoding='utf-8'))
    if config_hash(old)!=old['config_sha256']: raise RuntimeError('Existing locked config hash is invalid')
    new=normalized(old);new['config_sha256']=config_hash(new)
    if old!=new:
        ledger=dict(reason='Only dependency path separator normalization',old_config_sha256=old['config_sha256'],new_config_sha256=new['config_sha256'],
                    old_file_sha256=sha(LOCK),dependencies=new['dependencies'],scientific_configuration={k:v for k,v in new.items() if k not in ('dependencies','config_sha256')})
        # Write audit first so an interruption never loses the historical identity.
        atomic_json(AUDIT/'path_migration.json',ledger)
        atomic_json(LOCK,new)
        ledger['new_file_sha256']=sha(LOCK);atomic_json(AUDIT/'path_migration.json',ledger)
    atomic_json(AUDIT/'dependency_verification.json',dict(passed=True,checked=checked,entries=normalized(references()),config_sha256=new['config_sha256']))
    return new
def aliases(config):
    result={config['config_sha256']};p=AUDIT/'path_migration.json'
    if p.exists():
        d=json.loads(p.read_text(encoding='utf-8'))
        science={k:v for k,v in config.items() if k not in ('dependencies','config_sha256')}
        if d['new_config_sha256']==config['config_sha256'] and d['dependencies']==config['dependencies'] and json.dumps(d['scientific_configuration'],sort_keys=True)==json.dumps(science,sort_keys=True):
            result.add(d['old_config_sha256'])
    return result
if __name__=='__main__': print(json.dumps(migrate(),indent=2))
