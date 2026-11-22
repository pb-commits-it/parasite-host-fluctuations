"""Compile all corrected figures offline from the existing Tectonic cache.

The default Tectonic network client panics under this macOS sandbox. A temporary
directory bundle refers to the unchanged cached resources and their bundle ID.
No downloads, services, cache mutations or network permissions are needed.
"""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))
import os,subprocess,tempfile
from study_io import HERE,RESULTS,save,now


def run():
    cache=Path.home()/'Library/Caches/Tectonic/bundles/data'
    bundles=sorted(p for p in cache.iterdir() if p.is_dir() and len(p.name)==64)
    assert len(bundles)==1,'Set up an unambiguous cached TeX bundle before compiling'
    source=bundles[0];records=[]
    with tempfile.TemporaryDirectory(prefix='lpd-fix1-tex-') as tmp:
        bundle=Path(tmp)
        for p in source.iterdir():(bundle/p.name).symlink_to(p)
        (bundle/'SHA256SUM').write_text(source.name+'\n')
        for name in ['memory_map','acf_by_interval','regime_fractions','sensitivity_fractions']:
            cmd=['tectonic','-b',str(bundle),'--untrusted',name+'.tex']
            with (RESULTS/(name+'_tectonic_fix1_offline_output.txt')).open('w') as out:
                p=subprocess.run(cmd,cwd=HERE/'figures',stdout=out,stderr=subprocess.STDOUT)
            records.append(dict(figure=name,command=cmd,returncode=p.returncode,finished_utc=now(),bundle=source.name))
            save(RESULTS/'fix1_figure_compile_log.json',records)
            assert p.returncode==0,name
            print(name,'PDF complete',flush=True)


if __name__=='__main__':run()
