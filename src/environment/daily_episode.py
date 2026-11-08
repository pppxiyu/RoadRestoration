"""Coordinate repair, traffic, public observations and scoring at daily boundaries."""
from src.environment.daily_restoration import RestorationState
from src.environment.daily_observation import build_observation
from src.environment.daily_performance import RecoveryMonitor


class DailyEpisode:
    """The stable experiment/policy interface; each component owns its own state."""
    def __init__(self, simulator, scenario, behavior, *, disconnected_flow):
        self.simulator, self.scenario, self.behavior = simulator, scenario, behavior
        self.disconnected_flow = disconnected_flow
        self.restoration = RestorationState(scenario)
        self.performance = RecoveryMonitor(simulator.rules, simulator.ctx, disconnected_flow)
        self.traffic = None
        self.initial_loss = 0.0
        self._solve_today()
        self.initial_loss = self._advance_without_action()

    @property
    def day(self):
        return self.restoration.day

    @property
    def completed(self):
        return self.restoration.completed

    @property
    def starts(self):
        return self.restoration.starts

    @property
    def order(self):
        return self.restoration.order

    @property
    def records(self):
        return self.performance.records

    @property
    def total_loss(self):
        return self.performance.total_loss

    @property
    def restored_streak(self):
        return self.performance.restored_streak

    @property
    def finished(self):
        return self.performance.finished

    def _solve_today(self):
        self.traffic = self.simulator.solve(
            self.day, self.restoration.damage(), self.behavior, self.traffic)

    def _candidates(self):
        return self.restoration.candidates()

    def observation(self):
        if self.finished:
            return None
        return build_observation(self.day, self.restoration.visible(), self.completed,
                                 self._candidates(), self.traffic, self.simulator.ctx.get("edges"))

    def _charge_today(self):
        return self.performance.record(self.day, self.traffic,
            repaired=len(self.completed), total_damaged=len(self.restoration.truth),
            discovered=len(self.restoration.visible()))

    def _advance_without_action(self):
        cost = 0.0
        while not self.finished and not self._candidates():
            cost += self._charge_today()
            if not self.finished:
                self.restoration.advance_day()
                self._solve_today()
        return cost

    def step(self, road):
        if self.finished:
            raise ValueError("must select one currently discovered, unrepaired road; waiting is not an action")
        self.restoration.begin(road)
        cost = 0.0
        while self.restoration.active_road is not None:
            cost += self._charge_today()
            self.restoration.advance_day()
            self._solve_today()
        cost += self._advance_without_action()
        return self.observation(), -cost, self.finished

    def result(self):
        if not self.finished:
            raise RuntimeError("an unfinished episode is not an evaluation result")
        return dict(scenario=self.scenario.scenario_id, objective=self.total_loss,
                    time_loss=sum(r["time_loss"] for r in self.records),
                    flow_loss=sum(r["flow_loss"] for r in self.records),
                    recovery_day=self.day, repair_completion_day=max(self.completed.values()),
                    order=list(self.order), starts=dict(self.starts), completions=dict(self.completed),
                    daily=self.records, initial_loss=self.initial_loss)
