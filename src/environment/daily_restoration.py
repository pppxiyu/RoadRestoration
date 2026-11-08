"""Physical repair state and report availability; no traffic or loss calculations."""


class RestorationState:
    def __init__(self, scenario):
        self.scenario = scenario
        self.truth = {r.edge_id: r for r in scenario.roads}
        self.day = 1
        self.completed, self.starts = {}, {}
        self.order = []
        self.active_road = None
        self.completion_day = None

    def visible(self):
        return self.scenario.visible(self.day)

    def candidates(self):
        return tuple(sorted(r.edge_id for r in self.visible() if r.edge_id not in self.starts))

    def damage(self):
        return {e: r.true_severity for e, r in self.truth.items() if e not in self.completed}

    def begin(self, road):
        if self.active_road is not None or road not in self.candidates():
            raise ValueError("must select one currently discovered, unrepaired road; waiting is not an action")
        self.starts[road] = self.day
        self.order.append(road)
        self.active_road = road
        self.completion_day = self.day + self.truth[road].true_duration

    def advance_day(self):
        self.day += 1
        if self.active_road is not None and self.day == self.completion_day:
            self.completed[self.active_road] = self.day
            self.active_road = self.completion_day = None
