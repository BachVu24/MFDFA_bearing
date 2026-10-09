"""Portability, integrity, resume and native fallback regression checks."""
import ast, hashlib, json, os, shutil, unittest, uuid
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch
import numpy as np
from execution import provenance as V, runner as R, accelerator as A

@contextmanager
def scratch():
    p=V.AUDIT/'_test_work'/uuid.uuid4().hex
    p.mkdir(parents=True)
    try: yield str(p)
    finally:
        assert p.resolve().is_relative_to(V.AUDIT.resolve())
        shutil.rmtree(p)

class ExecutionTests(unittest.TestCase):
    def test_posix_and_windows_path_resolve_identically(self):
        self.assertEqual(V.resolve(r'code\core\vmd.py'),V.resolve('code/core/vmd.py'))
        for path in ['/etc/passwd','../DATA','D:/DATA']:
            with self.assertRaises(ValueError): V.resolve(path)
    def test_migration_is_idempotent_and_science_unchanged(self):
        before=V.sha(V.LOCK);config=V.migrate()
        self.assertEqual(before,V.sha(V.LOCK));self.assertEqual(V.config_hash(config),config['config_sha256'])
        self.assertTrue(all('\\' not in x['path'] for x in config['dependencies']))
        audit=json.loads((V.AUDIT/'path_migration.json').read_text(encoding='utf-8'))
        self.assertEqual(audit['scientific_configuration'],{k:v for k,v in config.items() if k not in ('dependencies','config_sha256')})
        self.assertEqual(len(V.aliases(config)),2)
    def test_missing_and_changed_dependency_stop(self):
        with scratch() as d:
            p=Path(d)/'payload';p.write_text('first');entry=dict(path=p.relative_to(V.ROOT).as_posix(),sha256=V.sha(p))
            V.verify([entry]);p.write_text('changed')
            with self.assertRaisesRegex(RuntimeError,'CHANGED'): V.verify([entry])
            p.unlink()
            with self.assertRaisesRegex(RuntimeError,'MISSING'): V.verify([entry])
    def test_checkpoint_rejects_partial_corrupt_or_wrong_config(self):
        with scratch() as d:
            p=Path(d)/'payload';p.write_text('result');check=Path(d)/'checkpoint.json'
            source={'sha256':'source','recording':'recording','samples':64};config={'config_sha256':'config'}
            p2=Path(d)/'payload2';p2.write_text('result2')
            obj=dict(schema=1,complete=True,recording='recording',samples=64,channels=['output_voltage','input_voltage'],source_sha256='source',config_sha256='config',execution_sha256=V.sha(Path(R.__file__)),files=[dict(path=x.relative_to(V.ROOT).as_posix(),sha256=V.sha(x)) for x in [p,p2]])
            self.assertFalse(R.checkpoint_valid(check,source,config));V.atomic_json(check,obj)
            self.assertTrue(R.checkpoint_valid(check,source,config));self.assertFalse(R.checkpoint_valid(check,source,{'config_sha256':'different'}))
            p.write_text('corrupt');self.assertFalse(R.checkpoint_valid(check,source,config));p.unlink();self.assertFalse(R.checkpoint_valid(check,source,config))
            check.write_text('{');self.assertFalse(R.checkpoint_valid(check,source,config))
    def test_linux_never_loads_windows_dll_when_no_compiler(self):
        with scratch() as d, patch.object(A.platform,'system',return_value='Linux'),patch.object(A.shutil,'which',return_value=None),patch.object(A.ctypes,'CDLL') as load:
            result=A.initialize(d);self.assertEqual(result['backend'],'numpy_reference');load.assert_not_called()
        x=np.arange(64,dtype=float)
        a=A.vmd(x,K=2,max_iter=10);b=A.reference_vmd(x,K=2,max_iter=10)
        for u,v in zip(a,b):np.testing.assert_array_equal(u,v)
    def test_native_numerical_parity_gate(self):
        with scratch() as d:
            result=A.initialize(d)
            if result['backend']=='cpp_fused_admm': self.assertTrue(result['parity_passed']);self.assertTrue(all(x['passed'] for x in result['parity_cases']))
            else: self.assertFalse(result['parity_passed']);self.assertIsNone(A.namespace['LIB'])
    def test_drive_payload_hashes_and_interruption_recovery(self):
        with scratch() as d:
            src=Path(d)/'source';dst=Path(d)/'drive';src.mkdir();(src/'arrays').mkdir();(src/'checkpoints').mkdir()
            (src/'arrays/a').write_text('signal');(src/'checkpoints/a.json').write_text('complete')
            R.sync_tree(src,dst);self.assertEqual(V.sha(src/'arrays/a'),V.sha(dst/'arrays/a'))
            (dst/'arrays/a').write_text('interrupted');R.sync_tree(src,dst)
            self.assertEqual(V.sha(src/'arrays/a'),V.sha(dst/'arrays/a'));self.assertFalse(list(dst.rglob('*.tmp')))
    def test_compile_failure_falls_back_before_library_load(self):
        import subprocess
        with scratch() as d,patch.object(A.platform,'system',return_value='Linux'),patch.object(A.shutil,'which',return_value='/fake/g++'),patch.object(A.subprocess,'run',return_value=subprocess.CompletedProcess([],1,'','compile failed')),patch.object(A.ctypes,'CDLL') as load:
            result=A.initialize(d);self.assertEqual(result['backend'],'numpy_reference');load.assert_not_called()
    def test_failed_numerical_parity_disables_native_backend(self):
        if A.platform.system()!='Windows': self.skipTest('Uses existing Windows library for failure injection')
        def wrong(*args,**kwargs):
            result=A.reference_vmd(*args,**kwargs)
            return (result[0]+.01,*result[1:])
        with scratch() as d,patch.object(A,'_native',side_effect=wrong):
            result=A.initialize(d);self.assertEqual(result['backend'],'numpy_reference');self.assertIsNone(A.namespace['LIB'])
    def test_validated_recording_reuse_and_resume_without_computation(self):
        output=V.ROOT/'result/test/colab_runs/step5/resume_check'
        if not output.exists(): self.skipTest('Local recording resume integration fixture absent')
        P=R.pipeline(output);config=P.lock_config()
        source=json.loads((P.HISTORICAL_BASE/'config/source_inventory.json').read_text(encoding='utf-8'))[0]
        for c in P.CHANNELS:
            obj=R.historical_channel(P,source,c[1],config)
            self.assertIsNotNone(obj);V.verify(obj['array_hashes'])
        # Only metadata/array verification, never call a numerical pipeline.
        with patch.object(P,'inventory',return_value=(config,[source])),patch.object(P,'signal_job',side_effect=AssertionError('Unexpected recomputation')),patch.object(P.accelerator,'initialize',return_value={'backend':'verified-test'}):
            R.run('step5','resume_check')
        self.assertTrue(R.checkpoint_valid(output/'checkpoints'/(source['recording']+'.json'),source,config))
    def test_notebook_and_all_python_sources_parse(self):
        n=json.loads((V.ROOT/'colab_runner.ipynb').read_text(encoding='utf-8'));self.assertEqual(n['nbformat'],4)
        for c in n['cells']:
            if c['cell_type']=='code':ast.parse(''.join(c['source']))
        for p in V.ROOT.rglob('*.py'):
            if '__pycache__' not in p.parts:ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p))
    def test_validated_output_directory_is_protected(self):
        with self.assertRaises(ValueError):R.pipeline(V.ROOT/'result/test/step5')
if __name__=='__main__':
    import io,warnings
    stream=io.StringIO()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ExecutionTests))
    print(stream.getvalue())
    V.atomic_json(V.AUDIT/'execution_tests.json',dict(tests_run=result.testsRun,passed=result.wasSuccessful(),failures=len(result.failures),errors=len(result.errors)))
    (V.AUDIT/'execution_tests.log').write_text(stream.getvalue(),encoding='utf-8')
    raise SystemExit(0 if result.wasSuccessful() else 1)
