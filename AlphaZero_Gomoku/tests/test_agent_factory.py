"""P0.2: shared agent factory, az_v1 feature flags and Renju safety of the three AZ milestones."""

from pathlib import Path

import pytest
import torch

from az.model import AZNet, save_checkpoint
from az.player_tactical import TacticalAZPlayer
from az.threat_space import ThreatSpaceAZPlayer
from config import DEFAULT_CONFIG, load_config
from gomoku.board import BLACK, SIZE, WHITE, Board
from gomoku.rules import RenjuRules, get_rules
from gomoku.runner import Runner
from players.az_player import AZPlayer
from players.factory import create_player
from players.random import RandomPlayer

CONFIGS = Path(__file__).resolve().parent.parent / "configs"
AZ_VARIANTS = ("az_v0", "az_v1_tactical", "az_v1_threat")


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory):
    """Random-weight network: only exercises loading and wiring, never strength."""
    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("ckpt") / "tiny.pt"
    save_checkpoint(AZNet(n_blocks=1, n_filters=8), path, meta={"rule_set": "normal"})
    return path


def board_with(black=(), white=(), to_move=BLACK) -> Board:
    board = Board()
    for r, c in black:
        board.grid[r * SIZE + c] = BLACK
    for r, c in white:
        board.grid[r * SIZE + c] = WHITE
    board.current_player = to_move
    return board


def base_of(player):
    return getattr(player, "base", player)


# -- config flags ------------------------------------------------------------


def test_feature_flags_default_to_false():
    assert DEFAULT_CONFIG["tactical"] == {"enabled": False}
    assert DEFAULT_CONFIG["threat_space"] == {"enabled": False}
    for name in ("normal.yaml", "renju.yaml", "normal_5000.yaml"):
        cfg = load_config(CONFIGS / name)
        assert cfg["tactical"]["enabled"] is False, name
        assert cfg["threat_space"]["enabled"] is False, name


def test_load_config_does_not_share_nested_sections():
    first = load_config()
    first["tactical"]["enabled"] = True
    assert load_config()["tactical"]["enabled"] is False
    assert DEFAULT_CONFIG["tactical"]["enabled"] is False


# -- the three milestones ----------------------------------------------------


def test_factory_builds_three_milestones_from_one_checkpoint(tiny_checkpoint):
    rules = get_rules("normal")
    players = {
        name: create_player(name, rules, checkpoint=tiny_checkpoint, seed=7, n_simulations=16)
        for name in AZ_VARIANTS
    }

    assert type(players["az_v0"]) is AZPlayer
    assert type(players["az_v1_tactical"]) is TacticalAZPlayer
    assert players["az_v1_tactical"].enabled is True
    assert type(players["az_v1_threat"]) is ThreatSpaceAZPlayer
    assert players["az_v1_threat"].enabled is True
    assert players["az_v1_threat"].threat_enabled is True

    reference = base_of(players["az_v0"]).model.state_dict()
    for name, player in players.items():
        base = base_of(player)
        assert base.cfg["n_simulations"] == 16, name
        state = base.model.state_dict()
        assert all(torch.equal(reference[k], state[k]) for k in reference), name


def test_n_simulations_override_is_optional(tiny_checkpoint):
    rules = get_rules("normal")
    default = create_player("az_v0", rules, checkpoint=tiny_checkpoint)
    assert default.cfg["n_simulations"] == load_config(CONFIGS / "normal.yaml")["n_simulations"]
    with pytest.raises(ValueError):
        create_player("az_v0", rules, checkpoint=tiny_checkpoint, n_simulations=0)


def _play_sequence(player, rules, n_moves=6):
    """Let `player` choose both sides for a few moves; return the chosen moves."""
    board = Board()
    moves = []
    for _ in range(n_moves):
        move = player.select_move(board)
        moves.append(move)
        board.play(*move)
        if rules.is_terminal(board):
            break
    return moves


def test_flags_off_reproduce_az_v0_with_same_seed(tiny_checkpoint):
    rules = get_rules("normal")
    cfg = load_config(CONFIGS / "normal.yaml")
    cfg["n_simulations"] = 16

    direct = AZPlayer(tiny_checkpoint, rules, cfg=cfg, seed=3)
    via_factory = create_player("az_v0", rules, checkpoint=tiny_checkpoint, seed=3, n_simulations=16)
    disabled_wrapper = TacticalAZPlayer(AZPlayer(tiny_checkpoint, rules, cfg=cfg, seed=3), cfg=cfg)
    disabled_threat = ThreatSpaceAZPlayer(AZPlayer(tiny_checkpoint, rules, cfg=cfg, seed=3), cfg=cfg)

    expected = _play_sequence(direct, rules)
    assert _play_sequence(via_factory, rules) == expected
    assert _play_sequence(disabled_wrapper, rules) == expected
    assert _play_sequence(disabled_threat, rules) == expected
    assert disabled_wrapper.last_override_applied is False


def test_tactical_matches_az_v0_when_no_tactic_exists(tiny_checkpoint):
    rules = get_rules("normal")
    board = board_with(black=[(5, 5)], white=[(4, 4)])
    v0 = create_player("az_v0", rules, checkpoint=tiny_checkpoint, seed=11, n_simulations=16)
    v1 = create_player("az_v1_tactical", rules, checkpoint=tiny_checkpoint, seed=11, n_simulations=16)

    assert v1.select_move(board.copy()) == v0.select_move(board.copy())
    assert v1.last_override_applied is False


# -- Renju: no forbidden move from any milestone -----------------------------


@pytest.mark.parametrize("name", AZ_VARIANTS)
@pytest.mark.parametrize(
    "stones",
    [
        # (5, 5) would make an overline (six) for Black.
        {"black": [(5, c) for c in (1, 2, 3, 4, 6)], "white": [(0, 0), (0, 2), (10, 10), (10, 8), (0, 10)]},
        # (5, 5) would make a double three for Black.
        {"black": [(5, 4), (5, 6), (4, 5), (6, 5)], "white": [(0, 0), (0, 2), (10, 10), (10, 8)]},
    ],
)
def test_az_milestones_never_choose_renju_forbidden_moves(tiny_checkpoint, name, stones):
    rules = RenjuRules()
    board = board_with(stones["black"], stones["white"], to_move=BLACK)
    forbidden = set(rules.forbidden_moves(board))
    assert 5 * SIZE + 5 in forbidden

    player = create_player(name, rules, checkpoint=tiny_checkpoint, seed=0, n_simulations=32)
    before = board.grid.copy()
    r, c = player.select_move(board)

    assert r * SIZE + c in rules.legal_moves(board)
    assert r * SIZE + c not in forbidden
    assert (board.grid == before).all()


@pytest.mark.parametrize("name", AZ_VARIANTS)
def test_az_milestones_finish_renju_games_without_illegal_moves(tiny_checkpoint, name):
    rules = RenjuRules()
    for game in range(2):
        az = create_player(name, rules, checkpoint=tiny_checkpoint, seed=game, n_simulations=4)
        opponent = RandomPlayer(rules, seed=100 + game)
        black, white = (az, opponent) if game % 2 == 0 else (opponent, az)
        # Runner raises IllegalMoveError on any move outside rules.legal_moves.
        result = Runner(rules).play(black, white)
        assert result.moves
