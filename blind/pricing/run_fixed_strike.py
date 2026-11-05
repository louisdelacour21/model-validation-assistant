import sys, importlib.util, json, math
from valkit import bench
from valkit.campaign import pricing_checks
from valkit.findings import opinion
d=sys.argv[1]
s=importlib.util.spec_from_file_location(d,f"blind/pricing/{d}/model.py");m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class Fixed:
    """Strike stays fixed when spot is bumped: re-express it as k' = k + ln(100/spot)."""
    def __init__(s): s.m=m.Model(); s.name=s.m.name; s.domain=s.m.domain
    def price(s,p,spot=None):
        if spot is None: return s.m.price(p)
        return s.m.price({**p,"k":p["k"]+math.log(100.0/spot)},spot)
res=pricing_checks(Fixed())
out={"opinion":opinion(res),"verdict":bench.verdict(res),"status":{x.id:x.status for x in res}}
json.dump(out,open(f"blind/pricing/{d}/result_fixed_strike.json","w"))
print(d,out,flush=True)
