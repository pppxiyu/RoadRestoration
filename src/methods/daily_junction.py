"""Public OD observations -> shared junction embeddings -> road-node S2V.

Replay contains raw normalized observations, never learned embeddings. The
shared OD and junction encoders are part of the online/target Q-network and
are optimized and checkpointed with it. No realized damage or response rate
enters this module.
"""
import numpy as np
import pandas as pd

from src.environment.poi_response import CATEGORIES, COUNTS_PATH
from src.methods.daily_s2v import DailyEncoder, GLOBAL_INPUTS


class JunctionEncoder:
    node_dimension = 32
    global_dimension = 7

    def __init__(self, problem, context, base_gamma, *, demand_memory=False, bidirectional=False):
        self.demand_memory = bool(demand_memory)
        self.bidirectional = bool(bidirectional)
        if self.bidirectional and not self.demand_memory:
            raise ValueError("dual origin/destination encoding requires the approved nine OD inputs")
        self.raw_od_dimension = 9 if self.demand_memory else 6
        self.latent_dimension = 16 if self.demand_memory else 8
        self.hidden_dimension = 2 * self.latent_dimension
        self.node_dimension = 16 + 2 * self.latent_dimension
        self.base = DailyEncoder(problem, context, road_class_input=True)
        self.rules = problem["rules"]
        self.demand_scale = max(float(self.base.H0.max()), 1.0)
        self.ids, self.A, self.deg = self.base.ids, self.base.A, self.base.deg
        self.zone_ids = tuple(sorted(int(z) for z in context["zone_ids"]))
        self.od_pairs = tuple((int(o), int(d)) for o, d in context["od_pairs"])
        position = {z: i for i, z in enumerate(self.zone_ids)}
        self.od_indices = np.array([(position[o], position[d]) for o, d in self.od_pairs])
        self.valid = np.zeros((len(position), len(position)), dtype=bool)
        self.valid[tuple(self.od_indices.T)] = True
        if np.diag(self.valid).any() or len(self.od_pairs) != int(self.valid.sum()):
            raise ValueError("OD pairs must be unique and exclude self trips")
        table = pd.read_csv(COUNTS_PATH).set_index("od_node")
        counts = table.loc[list(self.zone_ids), list(CATEGORIES)].to_numpy(dtype=np.float32)
        self.poi_count_scale = 165.0
        if not np.isfinite(counts).all() or (counts < 0).any() or counts.max() > self.poi_count_scale:
            raise ValueError("facility counts violate the approved public normalization")
        self.template = np.zeros((*self.valid.shape, self.raw_od_dimension), dtype=np.float32)
        # The second index is always the destination, for every origin.
        self.template[:, :, 1:6] = counts[None, :, :] / self.poi_count_scale
        self.template[~self.valid] = 0
        rows = problem["edges"].set_index("edge_id").loc[list(self.ids)]
        self.endpoints = np.array([(position[int(r.u)], position[int(r.v)])
                                  for r in rows.itertuples()], dtype=np.int64)
        self.base_gamma = float(base_gamma)
        if not np.isfinite(self.base_gamma) or self.base_gamma <= 0:
            raise ValueError("base travel-time sensitivity must be public and positive")

    def encode(self, observation):
        state = self.base.encode(observation)
        history = np.asarray(observation.cumulative_unmet, dtype=float)
        if history.shape != (len(self.od_pairs),) or not np.isfinite(history).all() or (history < 0).any():
            raise ValueError("one finite nonnegative observed history is required per valid OD")
        state["x"] = np.delete(state["x"], 9, axis=1)
        state["g"] = np.append(state["g"], np.float32(self.base_gamma)).astype(np.float32)
        state["od"] = self.template.copy()
        state["od"][self.od_indices[:, 0], self.od_indices[:, 1], 0] = history / 35.0
        if self.demand_memory:
            reference = np.asarray(observation.external, dtype=float)
            deviation = np.asarray(observation.previous_observed_deviation, dtype=float)
            reachable = np.asarray(observation.previous_reachable)
            for value in (reference, deviation, reachable):
                if value.shape != history.shape or not np.isfinite(value).all():
                    raise ValueError("invalid public OD demand-memory observation")
            if (reference < 0).any() or not np.isin(reachable, [False, True]).all():
                raise ValueError("invalid reference demand or reachability")
            oi, di = self.od_indices.T
            state["od"][oi, di, 6] = reference / self.demand_scale
            state["od"][oi, di, 7] = deviation / self.demand_scale
            state["od"][oi, di, 8] = reachable
        return state

    def specification(self):
        old = self.base.specification()
        spec = dict(version="junction_od8_v1", road_ids=self.ids,
            raw_road_inputs=[name for i, name in enumerate(old["node_inputs"]) if i != 9],
            raw_road_dimension=16, node_dimension=32, node_count=len(self.ids),
            global_inputs=GLOBAL_INPUTS + ("public base travel-time sensitivity, per minute",),
            global_dimension=7, base_gamma=self.base_gamma, zone_ids=self.zone_ids,
            valid_od_pairs=self.od_pairs, valid_od_mask=self.valid.tolist(),
            road_endpoint_indices=self.endpoints.tolist(), destination_categories=CATEGORIES,
            raw_od_shape=[len(self.zone_ids), len(self.zone_ids), 6],
            raw_od_inputs=["cumulative unmet normal-demand ratio through yesterday / 35"] +
                          ["destination " + name + " facility count / 165" for name in CATEGORIES],
            od_encoder=[6, 16, 8], junction_encoder=[len(self.zone_ids) * 8, 16, 8],
            activation="ReLU only between the two linear layers; final embeddings signed",
            masking="self and nonexistent OD zero before and after shared OD encoder; valid zero history retained",
            aggregation="fixed destination slots flattened, not summed across OD",
            endpoints="concatenate 8-dimensional sum and 8-dimensional absolute difference",
            replay="raw x, g and OD observations; embeddings recomputed with current network weights",
            time_scale_days=35, poi_count_scale=165,
            information_boundary="public network/facility counts/base gamma and observed past trips only; no true damage, future events, gamma-prime or decay rates",
            optimization="all encoder and S2V parameters jointly learned with unchanged Q-learning losses")
        if self.demand_memory:
            spec.update(version="junction_od16_demand_memory_v2", node_dimension=self.node_dimension,
                raw_od_shape=[len(self.zone_ids), len(self.zone_ids), self.raw_od_dimension],
                raw_od_inputs=spec["raw_od_inputs"] + [
                    "today's natural-recovery reference OD demand / common public demand scale",
                    "yesterday's observed served OD minus yesterday's reference OD / common public demand scale",
                    "yesterday's OD reachable (day one: known zero-history initialization marked valid)"],
                demand_scale=self.demand_scale, demand_scale_definition="maximum normal-period OD demand, shared by all OD",
                od_encoder=[9, 32, 16], junction_encoder=[len(self.zone_ids) * 16, 32, 16],
                endpoints="concatenate 16-dimensional sum and 16-dimensional absolute difference",
                initial_memory="zero deviation, valid history, matching the environment's zero-history start",
                history_information_boundary="observed served demand only; no hidden internal demand for disconnected OD",
                future_reference_window=False)
        if self.bidirectional:
            spec.update(version="junction_od16_memory_origin_destination_v1",
                aggregation="shared OD encoding; fixed destination slots per origin and fixed origin slots per destination; separate role encoders then concat and fuse",
                incoming_junction_encoder=[len(self.zone_ids) * 16, 32, 16],
                role_fusion_encoder=[32, 32, 16],
                traffic_conditioned_od_encoding=False)
        return spec

    def wrap_network(self, core):
        import torch
        from torch import nn

        class JunctionQNet(nn.Module):
            def __init__(network):
                super().__init__()
                network.core = core
                network.od_encoder = nn.Sequential(nn.Linear(self.raw_od_dimension, self.hidden_dimension),
                    nn.ReLU(), nn.Linear(self.hidden_dimension, self.latent_dimension))
                network.junction_encoder = nn.Sequential(
                    nn.Linear(len(self.zone_ids) * self.latent_dimension, self.hidden_dimension),
                    nn.ReLU(), nn.Linear(self.hidden_dimension, self.latent_dimension))
                if self.bidirectional:
                    network.incoming_encoder = nn.Sequential(
                        nn.Linear(len(self.zone_ids) * self.latent_dimension, self.hidden_dimension),
                        nn.ReLU(), nn.Linear(self.hidden_dimension, self.latent_dimension))
                    network.role_fusion = nn.Sequential(
                        nn.Linear(2 * self.latent_dimension, self.hidden_dimension),
                        nn.ReLU(), nn.Linear(self.hidden_dimension, self.latent_dimension))
                network.register_buffer("valid_od", torch.from_numpy(self.valid.copy()))
                network.register_buffer("endpoints", torch.from_numpy(self.endpoints.copy()))

            def junction_embeddings(network, od):
                # Mask both sides of the encoder: input padding must not enter
                # the network, and encoder bias must not make padding nonzero.
                od = od.masked_fill(~network.valid_od.unsqueeze(-1), 0.0)
                encoded = network.od_encoder(od)
                encoded = encoded.masked_fill(~network.valid_od.unsqueeze(-1), 0.0)
                outgoing = network.junction_encoder(encoded.flatten(start_dim=1))
                if not self.bidirectional:
                    return outgoing
                # encoded[o,d] stays paired with destination d's facilities.
                # Transposition only groups the same OD embeddings by destination.
                incoming = network.incoming_encoder(encoded.transpose(0, 1).flatten(start_dim=1))
                return network.role_fusion(torch.cat((outgoing, incoming), dim=-1))

            def road_features(network, x, od):
                junctions = network.junction_embeddings(od)
                left, right = junctions[network.endpoints[:, 0]], junctions[network.endpoints[:, 1]]
                return torch.cat((x, left + right, torch.abs(left - right)), dim=1)

            def forward(network, x, A, deg, g, od):
                return network.core(network.road_features(x, od), A, deg, g)

        return JunctionQNet()
