from types import SimpleNamespace

import scripts.play as play_cli
from gomoku.board import EMPTY
from gomoku.rules import get_rules
from players.alphabeta import AlphaBetaPlayer
from players.az_player import AZPlayer
from players.factory import create_player
from players.heuristic import HeuristicPlayer
from az.player_tactical import TacticalAZPlayer
from az.threat_space import ThreatSpaceAZPlayer


def test_cli_builds_distinct_selected_agents_for_each_side(monkeypatch):
    observed = {}

    class StubRunner:
        def __init__(self, _rules):
            pass

        def play(self, black, white, *, on_move=None):
            observed["players"] = (black, white)
            return SimpleNamespace(reason="draw", winner=EMPTY, moves=[])

    monkeypatch.setattr(play_cli, "Runner", StubRunner)

    assert play_cli.main(
        [
            "--mode", "ai-vs-ai",
            "--black-agent", "heuristic",
            "--white-agent", "alphabeta",
            "--depth", "1",
        ]
    ) == 0

    black, white = observed["players"]
    assert isinstance(black, HeuristicPlayer)
    assert isinstance(white, AlphaBetaPlayer)
    assert white.depth == 1
    assert black is not white


def test_az_variants_use_the_shared_checkpoint_factory(monkeypatch):
    class StubAZPlayer:
        def __init__(self, _checkpoint, rules, cfg, **_kwargs):
            self.rules = rules
            self.cfg = cfg

    monkeypatch.setattr(AZPlayer, "__init__", StubAZPlayer.__init__)
    rules = get_rules("normal")

    tactical = create_player("az_v1_tactical", rules, checkpoint="fake.pt")
    threat = create_player("az_v1_threat", rules, checkpoint="fake.pt")

    assert isinstance(tactical, TacticalAZPlayer)
    assert tactical.enabled
    assert isinstance(threat, ThreatSpaceAZPlayer)
    assert threat.enabled
    assert threat.threat_enabled
