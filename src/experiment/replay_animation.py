"""Relocate the frozen replay entry in memory; keep its scientific code intact."""
from pathlib import Path
import hashlib,json,sys
ROOT=Path(__file__).resolve().parent
for rel,h in json.loads((ROOT/'prepared_manifest.json').read_text()).items():
    assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==h,rel
source=ROOT/'historical_replay_cases.py'
text=source.read_text(encoding='utf8')
old="ROOT=Path('/scratch/xp2239/demand_response_decision_ranking_v3_anchor_20261103')"
assert text.count(old)==1
text=text.replace(old,'ROOT=Path(__file__).resolve().parent')
sys.path.insert(0,str(ROOT))
exec(compile(text,str(source),'exec'),dict(__name__='__main__',__file__=str(source)))
