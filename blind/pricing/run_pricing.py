import sys, importlib.util, json
from valkit import bench
from valkit.campaign import pricing_checks
from valkit.findings import opinion
d=sys.argv[1]
s=importlib.util.spec_from_file_location(d,f"blind/pricing/{d}/model.py");m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
res=pricing_checks(m.Model())
out={"opinion":opinion(res),"verdict":bench.verdict(res),"status":{x.id:x.status for x in res}}
json.dump(out,open(f"blind/pricing/{d}/result.json","w"))
print(d,out,flush=True)
