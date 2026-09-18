"""
Unit Tests for Nim Game AI
===========================
Tests verify:
  1. Nim-sum and Grundy values calculation
  2. N-position vs P-position classification
  3. Optimal move generation guarantees new Nim-sum = 0
  4. Misère play rule handling
  5. Minimax search equivalence with Bouton's Theorem
  6. Optimal AI 100% win rate against Random AI from winning positions
"""

try:
    import pytest
except ImportError:
    pytest = None
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from app import (
    GameState,
    calculate_mex,
    grundy_value,
    minimax_search,
    get_optimal_move,
    get_random_move,
    simulate_match,
    run_benchmark,
)


def test_calculate_mex():
    assert calculate_mex(set()) == 0
    assert calculate_mex({1, 2, 3}) == 0
    assert calculate_mex({0, 1, 3}) == 2
    assert calculate_mex({0, 1, 2, 4}) == 3


def test_nim_sum_and_grundy():
    # P-positions (Nim-sum == 0)
    assert grundy_value((1, 2, 3)) == 0
    assert grundy_value((2, 5, 7)) == 0
    assert grundy_value((1, 4, 5)) == 0

    # N-positions (Nim-sum != 0)
    assert grundy_value((3, 4, 5)) == 2
    assert grundy_value((1, 3, 5, 7)) == 0  # Standard 1,3,5,7 is a P-position!
    assert grundy_value((1, 3, 5)) == 7


def test_game_state_properties():
    # N-position
    state_n = GameState(heaps=[3, 4, 5], rule="Normal")
    assert state_n.nim_sum == 2
    assert state_n.is_winning_position is True
    assert state_n.is_game_over is False

    # P-position
    state_p = GameState(heaps=[1, 2, 3], rule="Normal")
    assert state_p.nim_sum == 0
    assert state_p.is_winning_position is False


def test_optimal_move_normal_play():
    # From [3, 4, 5], Nim-sum is 2 (binary 10).
    # Heaps are: 3 (011), 4 (100), 5 (101).
    # Bit 2 (value 2) is set in 3 and 5.
    # 3 ^ 2 = 1 (< 3): can reduce Heap 1 from 3 -> 1 (remove 2).
    # 5 ^ 2 = 7 (> 5): not a valid reduction.
    # 4 ^ 2 = 6 (> 4): not a valid reduction.
    # So Heap 1 (index 0) must be reduced by 2.
    heap_idx, remove_amt, expl = get_optimal_move([3, 4, 5], rule="Normal")
    assert heap_idx == 0
    assert remove_amt == 2

    # Verify after this move, new Nim-sum is 0
    new_heaps = [3, 4, 5]
    new_heaps[heap_idx] -= remove_amt
    assert (new_heaps[0] ^ new_heaps[1] ^ new_heaps[2]) == 0


def test_optimal_moves_always_restore_zero_nim_sum():
    # Only N-positions (Nim-sum != 0) have moves that restore Nim-sum to 0
    test_cases = [
        [3, 4, 5],       # Nim-sum = 2
        [1, 3, 5],       # Nim-sum = 7
        [5, 6, 7],       # Nim-sum = 4
        [10, 20, 25],    # Nim-sum = 11
        [4, 8, 12, 16],  # Nim-sum = 16
    ]
    for heaps in test_cases:
        heap_idx, remove_amt, _ = get_optimal_move(heaps, rule="Normal")
        assert remove_amt >= 1
        assert remove_amt <= heaps[heap_idx]
        after = list(heaps)
        after[heap_idx] -= remove_amt
        new_nim_sum = 0
        for h in after:
            new_nim_sum ^= h
        assert new_nim_sum == 0, f"Failed on {heaps}: resulting heaps {after} has sum {new_nim_sum}"


def test_p_position_defensive_move():
    # From P-positions [1, 2, 3] and [10, 20, 30], no move can leave Nim-sum = 0
    p_cases = [[1, 2, 3], [10, 20, 30], [2, 5, 7]]
    for heaps in p_cases:
        heap_idx, remove_amt, expl = get_optimal_move(heaps, rule="Normal")
        assert "P-position" in expl
        assert remove_amt >= 1
        assert remove_amt <= heaps[heap_idx]


def test_minimax_matches_bouton():
    # Small states where minimax is fast
    small_cases = [
        ((1, 2, 3), -1),  # P-position -> -1 (losing for mover)
        ((2, 2), -1),     # P-position
        ((1, 1), -1),     # P-position
        ((3, 4, 5), 1),   # N-position -> +1 (winning for mover)
        ((1, 2), 1),      # N-position
        ((0, 5), 1),      # N-position
    ]
    for heaps, expected in small_cases:
        assert minimax_search(heaps, True, "Normal") == expected


def test_misere_endgame_logic():
    # When exactly one heap > 1 and others are 1:
    # State: [3, 1, 1]. Other 1s count = 2 (even).
    # To leave an odd number of 1s, reduce 3 to 1 (leaving three 1s).
    idx, remove_amt, _ = get_optimal_move([3, 1, 1], rule="Misère")
    assert idx == 0
    assert remove_amt == 2  # leaves 1

    # State: [3, 1]. Other 1s count = 1 (odd).
    # To leave an odd number of 1s, reduce 3 to 0 (leaving one 1).
    idx, remove_amt, _ = get_optimal_move([3, 1], rule="Misère")
    assert idx == 0
    assert remove_amt == 3  # leaves 0


def test_random_move_validity():
    for _ in range(50):
        heaps = [3, 4, 5]
        idx, count = get_random_move(heaps)
        assert 0 <= idx < 3
        assert 1 <= count <= heaps[idx]


def test_optimal_ai_wins_100_percent_against_random():
    # Starting from winning position [3, 4, 5] (N-position, Nim-sum = 2)
    # Player 1 (Optimal AI) MUST win 100% of the games against Player 2 (Random)
    results = run_benchmark(
        initial_heaps=[3, 4, 5],
        num_games=50,
        rule="Normal",
        first_player_type="Optimal",
        second_player_type="Random",
    )
    assert results["p1_win_rate"] == 100.0
    assert results["p1_wins"] == 50
    assert results["p2_wins"] == 0


def test_second_player_optimal_wins_from_p_position():
    # Starting from balanced position [1, 2, 3] (P-position, Nim-sum = 0)
    # Player 1 (Random) must move first into an N-position.
    # Player 2 (Optimal AI) MUST win 100% of the games!
    results = run_benchmark(
        initial_heaps=[1, 2, 3],
        num_games=50,
        rule="Normal",
        first_player_type="Random",
        second_player_type="Optimal",
    )
    assert results["p2_win_rate"] == 100.0
    assert results["p2_wins"] == 50
    assert results["p1_wins"] == 0


if __name__ == "__main__":
    print("=" * 60)
    print("Running Nim Game AI Test Suite")
    print("=" * 60)
    tests = [
        test_calculate_mex,
        test_nim_sum_and_grundy,
        test_game_state_properties,
        test_optimal_move_normal_play,
        test_optimal_moves_always_restore_zero_nim_sum,
        test_p_position_defensive_move,
        test_minimax_matches_bouton,
        test_misere_endgame_logic,
        test_random_move_validity,
        test_optimal_ai_wins_100_percent_against_random,
        test_second_player_optimal_wins_from_p_position,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  [PASS] {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {t.__name__}: {e}")
            raise e
    print("=" * 60)
    print(f"All {passed}/{len(tests)} tests passed successfully!")
    print("=" * 60)
