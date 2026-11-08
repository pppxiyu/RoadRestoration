"""Rebuild the approved sandboxed animations from archived traffic data.

The standalone wrapper and CSP are preserved byte-for-byte outside the embedded
simulation payload. Orange trips mean positive OD demand relative to the same
action and date in the fixed-demand branch, passively assigned to response paths.
"""
from pathlib import Path
import gzip,html,json,re
from src import config

ROOT=Path(__file__).resolve().parents[2]

def main(kind, output):
    profile=config.results_profile('behavior_'+kind)
    folder=ROOT/profile['experiment']
    template=ROOT/profile['template']
    text=template.read_bytes().decode('utf8')
    iframe=re.search(r'data-srcdoc="([\s\S]*?)"',text)
    if not iframe:raise ValueError('Missing sandboxed iframe')
    assert 'sandbox="allow-scripts"' in text and 'Content-Security-Policy' in text
    inner=html.unescape(iframe[1])
    marker=",D="
    start=inner.index(marker)+len(marker)
    old,consumed=json.JSONDecoder().raw_decode(inner[start:])
    measured=json.loads(gzip.decompress((folder/f'data/visual_{kind}.json.gz').read_bytes()))
    # The published payload adds geometry and display choices. All scientific
    # data must exactly match the saved, verified replay, with no new simulation.
    for key,value in measured.items():
        if old.get(key)!=value:raise ValueError(f'Animation data mismatch: {kind}/{key}')
    new=dict(old,**measured)
    payload=json.dumps(new,separators=(',',':'),ensure_ascii=True,allow_nan=False)
    updated=inner[:start]+payload+inner[start+consumed:]
    # Reuse identical escaping when the data is unchanged, preserving the
    # exact approved display rather than substituting a raw fragment.
    if new==old:
        result=text
    else:
        result=text[:iframe.start(1)]+html.escape(updated,quote=True)+text[iframe.end(1):]
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    target=output/f'{kind}-response.html';target.write_bytes(result.encode('utf8'))
    return target
