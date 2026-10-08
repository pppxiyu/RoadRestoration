# Technical Note — Sioux Falls toy instance (Step 1)

Concise record of the two *fabricated* modeling choices in this step. Everything
else (topology, capacity/length/free-flow-time, BPR params, OD demand) is taken
unchanged from the public Sioux Falls files — see `README.md` for source details.

---

## 1. Road classification (`road_class`)

 The paper's level structure ([main.tex, Table 1](../../paper/main.tex)) needs each
segment's *road type* ∈ {local, major, highway}. The Sioux Falls `_net.tntp`
file does **not** carry functional class — its `link_type` column is uniformly
`1`. So `road_class` must be inferred.

**Logic.** Higher-capacity links carry the higher-order traffic function, so we
split the 38 undirected edges into three capacity bands (lower / middle / upper third):

| road_class | rule | capacity range (veh/h) | # edges |
|---|---|---|---|
| local   | capacity < q1            | 4824 – 5003   | 13 |
| major   | q1 ≤ capacity < q2       | 5046 – 9599   | 11 |
| highway | capacity ≥ q2            | 10000 – 25900 | 14 |

with cut-points **q1 = 5017**, **q2 = 10000** (the 33rd / 67th percentiles
of the capacity values).

---

## 2. Disruption instance (`disrupted_segments.csv`)

**How the 8 segments were chosen.** An arbitrary but reproducible toy choice
(seed = 42), not a constraint from the paper. The only enforced properties are that
all three road types appear (one edge of each is drawn first) and all three
severities appear (the severity rule fixes the counts at three 1s, three 2s, two
3s); the remaining edges are drawn uniformly at random. The specific edges and
severities below reproduce under the same seed. (The sampled set happens to include
two edges at the node-10 hub, 9–10 and 10–15, but that is an outcome, not a design
target.)

**Resulting instance.**

| edge_id | u–v | road_class (road type) | severity (`v_e*`) | level_id |
|---|---|---|---|---|
| 3  | 2–6   | local   | 1 | local-S1   |
| 7  | 4–11  | local   | 2 | local-S2   |
| 15 | 9–10  | highway | 1 | highway-S1 |
| 17 | 10–15 | highway | 3 | highway-S3 |
| 27 | 15–22 | major   | 3 | major-S3   |
| 31 | 18–20 | highway | 2 | highway-S2 |
| 34 | 20–22 | major   | 1 | major-S1   |
| 35 | 21–22 | major   | 2 | major-S2   |

The 8 segments fall on 8 distinct levels — all 9 possible `(road type, severity)`
levels except `local-S3` (highway and major each appear at all three severities,
local at S1 and S2).

---

## 3. Open-source data collected (used unchanged)

Everything in the toy **except** the two inferred items above — `road_class`
(Section 1) and the disruption instance (Section 2) — comes directly from the public
**Sioux Falls** benchmark, unchanged. **Source.** `bstabler/TransportationNetworks`, `SiouxFalls/` folder · The four downloaded files are kept untouched in
`raw/`.


| raw file | fields (columns) | provides | size |
|---|---|---|---|
| `SiouxFalls_net.tntp` | `init_node, term_node, capacity, length, free_flow_time, b (=BPR α), power (=BPR β), speed, toll, link_type` | link topology + capacity + length + free-flow time + BPR parameters (α = 0.15, β = 4 for every link) | 76 directed links |
| `SiouxFalls_node.tntp` | `Node, X, Y` | node coordinates (reproduce the standard layout) | 24 nodes |
| `SiouxFalls_trips.tntp` | origin-block: `Origin <o>` then `<dest> : <flow>;` pairs | full origin–destination demand matrix | 24 zones, 528 positive OD pairs, 360,600 trips total |
| `SiouxFalls_flow.tntp` | `From, To, Volume, Cost` | reference user-equilibrium (UE) solution. Per link: `Volume` = equilibrium flow; `Cost` = that link's travel time at that flow — the congested BPR value `free_flow_time × (1 + α(Volume/capacity)^β)`, so same units as `free_flow_time` (≥ free-flow time, larger when the link is busier). From these flows the UE **objective** — the Beckmann function (sum over links of the integral of link travel time from 0 to the link's flow) that UE assignment minimizes — equals its best-known optimum ≈ **4,231,335** (≈ 4.23×10⁶); the "42.31" sometimes quoted is just this value ÷ 10⁵. Kept to verify a future UE solver converges to this value — unused now | 76 links |

**Reformatted to CSV — parsing/reshaping only, no inference except where noted.**

| CSV | parsed from | columns | note |
|---|---|---|---|
| `network/nodes.csv` | `node.tntp` | `node_id, x, y` | unchanged |
| `network/od_pairs.csv` | `trips.tntp` | `od_id, origin, destination, h0` | unchanged; positive flows only (`h0` = pre-disaster OD demand) |
| `network/edges.csv` | `net.tntp` | `edge_id, u, v, capacity, length, free_flow_time, bpr_alpha, bpr_beta, road_class` | 76 directed → 38 undirected (Sioux Falls is symmetric — an exact merge, not an inference); every column unchanged **except `road_class`, which is inferred (Section 1)** |


**Design/supply inputs vs. assignment outputs.** A useful split of the quantities:
- **Design / supply inputs** — fixed network data, *not* the result of any traffic
  assignment. `capacity` is how many vehicles the link can carry per unit time,
  determined by the road's design (lanes, road type, speed) — it is a given, not a UE
  output. `free_flow_time` is the travel time at zero congestion (= length ÷
  free-flow speed) and serves as the BPR baseline t₀, the *lower bound* of a link's
  travel time; also a design input, not a UE output. `length` and the BPR shape
  parameters α, β are inputs too.
- **UE outputs** — produced by solving user equilibrium *on top of* those inputs:
  the per-link `Volume` and `Cost` in `flow.tntp`. In short, `capacity`/
  `free_flow_time` go *into* the BPR cost function `t = free_flow_time·(1+α(v/capacity)^β)`;
  `Volume`/`Cost` come *out* of the equilibrium.

**OD-demand time scope.** The files declare no time unit. The OD matrix is a single
*static* period: demand and capacity are consistent flow rates, by the usual Sioux
Falls convention read as **per hour** (and `free_flow_time` in minutes) — it is not a
full-day total. We use this matrix as the pre-disaster baseline demand `H^{t₀}`;
turning it into the paper's per-3 h-step dynamic demand is a later step.

