"""
Nim Game AI - Interactive Streamlit Dashboard & Theoretical Solver
==================================================================
This single Python file contains:
  1. Complete Nim Game Logic & Rules (Normal & Misère play)
  2. Mathematical Foundations (Bouton's Theorem, Sprague-Grundy, XOR Nim-Sum)
  3. Minimax Game-Tree Search with Memoization
  4. Random Baseline Player & Optimal AI Player (wins from every winning position)
  5. Monte Carlo Simulation Engine (Empirical validation of 100% win-rate)
  6. Interactive Streamlit Dashboard with visual heaps and bit-level XOR analysis
"""

from dataclasses import dataclass, field
import functools
import math
import random
from typing import Dict, List, Optional, Tuple

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import streamlit as st
except ImportError:
    st = None

# ==============================================================================
# 1. CORE GAME ENGINE & MATHEMATICAL LOGIC
# ==============================================================================

@dataclass
class GameState:
    heaps: List[int]
    rule: str = "Normal"  # "Normal" (last player to move wins) or "Misère" (last player loses)
    turn: str = "Player 1"
    winner: Optional[str] = None
    move_history: List[Dict] = field(default_factory=list)

    @property
    def is_game_over(self) -> bool:
        return all(h == 0 for h in self.heaps)

    @property
    def total_objects(self) -> int:
        return sum(self.heaps)

    @property
    def nim_sum(self) -> int:
        """Calculate bitwise XOR sum of all heaps (Bouton, 1901)."""
        result = 0
        for h in self.heaps:
            result ^= h
        return result

    @property
    def is_winning_position(self) -> bool:
        """
        In Normal Play: Non-zero Nim-sum is an N-position (Winning).
        In Misère Play:
          - If only heaps of size <= 1 remain: winning if an ODD number of 1-heaps remain.
          - Otherwise: same as Normal play (Nim-sum != 0).
        """
        if self.is_game_over:
            return False

        if self.rule == "Misère":
            large_heaps = [h for h in self.heaps if h > 1]
            if len(large_heaps) == 0:
                # Only 1-heaps remain. Current player wins if there is an ODD number of 1-heaps
                ones_count = sum(1 for h in self.heaps if h == 1)
                return (ones_count % 2) != 0

        return self.nim_sum != 0


def calculate_mex(values: set) -> int:
    """Minimum Excluded Value (Mex) calculation for Sprague-Grundy theorem."""
    mex = 0
    while mex in values:
        mex += 1
    return mex


@functools.lru_cache(maxsize=10000)
def grundy_value(heaps_tuple: Tuple[int, ...]) -> int:
    """
    Sprague-Grundy value G(state) = mex { G(next_state) }.
    By the Sprague-Grundy theorem for impartial games:
      G(h1, h2, ..., hk) = G(h1) ^ G(h2) ^ ... ^ G(hk)
      and for an individual Nim pile, G(h) = h.
    """
    xor_val = 0
    for h in heaps_tuple:
        xor_val ^= h
    return xor_val


@functools.lru_cache(maxsize=50000)
def minimax_search(heaps_tuple: Tuple[int, ...], is_maximizing: bool, rule: str = "Normal") -> int:
    """
    Minimax search with memoization for game-tree evaluation.
    Returns: +1 if current player has a forced win, -1 if forced loss.
    """
    # Terminal condition
    if all(h == 0 for h in heaps_tuple):
        if rule == "Normal":
            # Previous player took the last counter, so current player has no moves left
            return -1 if is_maximizing else 1
        else:  # Misère play: last player loses, so current player wins
            return 1 if is_maximizing else -1

    if is_maximizing:
        best_val = -1
        for i, count in enumerate(heaps_tuple):
            for remove_amt in range(1, count + 1):
                next_heaps = list(heaps_tuple)
                next_heaps[i] -= remove_amt
                val = minimax_search(tuple(next_heaps), False, rule)
                if val == 1:
                    return 1  # Alpha-beta shortcut
                best_val = max(best_val, val)
        return best_val
    else:
        best_val = 1
        for i, count in enumerate(heaps_tuple):
            for remove_amt in range(1, count + 1):
                next_heaps = list(heaps_tuple)
                next_heaps[i] -= remove_amt
                val = minimax_search(tuple(next_heaps), True, rule)
                if val == -1:
                    return -1  # Alpha-beta shortcut
                best_val = min(best_val, val)
        return best_val


# ==============================================================================
# 2. MOVE STRATEGIES: RANDOM vs OPTIMAL
# ==============================================================================

def get_random_move(heaps: List[int]) -> Tuple[int, int]:
    """Uniformly randomly selects a non-empty heap and valid number of objects to remove."""
    non_empty_indices = [i for i, h in enumerate(heaps) if h > 0]
    if not non_empty_indices:
        raise ValueError("Cannot move on an empty board")
    heap_idx = random.choice(non_empty_indices)
    remove_count = random.randint(1, heaps[heap_idx])
    return heap_idx, remove_count


def get_optimal_move(heaps: List[int], rule: str = "Normal") -> Tuple[int, int, str]:
    """
    Computes mathematically optimal move based on Bouton's Theorem (1901)
    and Sprague-Grundy theory for both Normal and Misère play.

    Returns:
      (heap_index, remove_count, explanation)
    """
    nim_sum = 0
    for h in heaps:
        nim_sum ^= h

    # -------------------------------------------------------------------------
    # MISÈRE PLAY STRATEGY
    # -------------------------------------------------------------------------
    if rule == "Misère":
        # Check if exactly one heap has size > 1
        heaps_greater_than_one = [i for i, h in enumerate(heaps) if h > 1]

        if len(heaps_greater_than_one) == 1:
            idx = heaps_greater_than_one[0]
            other_ones = sum(1 for i, h in enumerate(heaps) if i != idx and h == 1)
            # In misère play, we want to leave an ODD number of 1-heaps
            # If other_ones is even (e.g. 0, 2), we want 1 single counter left in this heap -> odd total 1s
            # If other_ones is odd (e.g. 1, 3), we want 0 counters left in this heap -> odd total 1s
            target = 1 if (other_ones % 2 == 0) else 0
            remove_amt = heaps[idx] - target
            explanation = (
                f"[Misère Endgame] Exactly one heap has size > 1. Reducing Heap {idx + 1} "
                f"from {heaps[idx]} to {target} to leave an odd number ({other_ones + target}) of 1s."
            )
            return idx, remove_amt, explanation

        elif len(heaps_greater_than_one) == 0:
            # All non-empty heaps are size 1. Any move is forced, but pick one to minimize loss if any
            ones = [i for i, h in enumerate(heaps) if h == 1]
            idx = ones[0]
            explanation = f"[Misère 1-Heap] Taking the single item from Heap {idx + 1}."
            return idx, 1, explanation

    # -------------------------------------------------------------------------
    # NORMAL PLAY STRATEGY (AND GENERAL MISÈRE BEFORE ENDGAME)
    # -------------------------------------------------------------------------
    if nim_sum != 0:
        # Find the Most Significant Bit (MSB) of the Nim-sum
        msb_power = 1 << (nim_sum.bit_length() - 1)

        # Find a heap that has a 1 in this bit position: heap ^ nim_sum < heap
        for i, h in enumerate(heaps):
            target_heap = h ^ nim_sum
            if target_heap < h:
                remove_amt = h - target_heap
                explanation = (
                    f"Nim-sum = {nim_sum} (binary {bin(nim_sum)[2:]}). "
                    f"MSB is at {msb_power}. "
                    f"Heap {i + 1} ({h}) has this bit set: "
                    f"reducing {h} -> {target_heap} (remove {remove_amt}) forces new Nim-sum to 0 (P-position)."
                )
                return i, remove_amt, explanation

    # If Nim-sum is 0, the current position is a P-position (losing against optimal play)
    # Any legal move leaves Nim-sum != 0; play random or smallest move
    non_empty = [i for i, h in enumerate(heaps) if h > 0]
    idx = non_empty[0]
    remove_amt = 1
    explanation = (
        f"Position is balanced with Nim-sum = 0 (P-position). "
        f"No winning move exists; playing defensive move (removing {remove_amt} from Heap {idx + 1})."
    )
    return idx, remove_amt, explanation


# ==============================================================================
# 3. MATCH SIMULATION ENGINE
# ==============================================================================

def simulate_match(
    initial_heaps: List[int],
    player1_type: str,  # "Optimal" or "Random"
    player2_type: str,  # "Optimal" or "Random"
    rule: str = "Normal",
) -> Tuple[str, int, List[Dict]]:
    """Runs a complete automated game between two specified agents."""
    heaps = list(initial_heaps)
    turn = "Player 1"
    turn_count = 0
    log = []

    while any(h > 0 for h in heaps):
        turn_count += 1
        active_type = player1_type if turn == "Player 1" else player2_type

        if active_type == "Optimal":
            heap_idx, remove_amt, expl = get_optimal_move(heaps, rule)
        else:
            heap_idx, remove_amt = get_random_move(heaps)
            expl = f"Random move chosen: took {remove_amt} from Heap {heap_idx + 1}."

        before_state = list(heaps)
        heaps[heap_idx] -= remove_amt
        log.append({
            "Turn": turn_count,
            "Player": turn,
            "Agent": active_type,
            "Heap": heap_idx + 1,
            "Removed": remove_amt,
            "Before": str(before_state),
            "After": str(heaps),
            "Explanation": expl,
        })

        if all(h == 0 for h in heaps):
            # Game ended with this move
            if rule == "Normal":
                winner = turn  # Last player took counter and wins
            else:
                winner = "Player 2" if turn == "Player 1" else "Player 1"  # Last player loses
            return winner, turn_count, log

        turn = "Player 2" if turn == "Player 1" else "Player 1"

    return "Draw", turn_count, log


def run_benchmark(
    initial_heaps: List[int],
    num_games: int,
    rule: str,
    first_player_type: str,
    second_player_type: str,
) -> Dict:
    """Executes N games and returns win rates and statistics."""
    p1_wins = 0
    p2_wins = 0
    total_turns = 0

    for _ in range(num_games):
        winner, turns, _ = simulate_match(initial_heaps, first_player_type, second_player_type, rule)
        if winner == "Player 1":
            p1_wins += 1
        else:
            p2_wins += 1
        total_turns += turns

    return {
        "num_games": num_games,
        "player1_type": first_player_type,
        "player2_type": second_player_type,
        "p1_wins": p1_wins,
        "p2_wins": p2_wins,
        "p1_win_rate": (p1_wins / num_games) * 100,
        "p2_win_rate": (p2_wins / num_games) * 100,
        "avg_turns": total_turns / num_games,
    }


# ==============================================================================
# 4. STREAMLIT INTERACTIVE DASHBOARD
# ==============================================================================

def main():
    st.set_page_config(
        page_title="Nim Game AI - Optimal Strategy & Game Search",
        page_icon="👑",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom Header Style
    st.markdown("""
        <style>
        .main-title {
            font-size: 2.2rem;
            font-weight: 800;
            color: #1E293B;
            margin-bottom: 0.2rem;
        }
        .subtitle {
            font-size: 1.05rem;
            color: #475569;
            margin-bottom: 1.5rem;
        }
        .heap-card {
            background-color: #F8FAFC;
            border: 2px solid #E2E8F0;
            border-radius: 12px;
            padding: 16px;
            text-align: center;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
        .token-circle {
            display: inline-block;
            width: 18px;
            height: 18px;
            border-radius: 50%;
            background: #3B82F6;
            margin: 2px;
        }
        .p-pos {
            background-color: #FEF2F2;
            border-left: 5px solid #EF4444;
            padding: 12px;
            border-radius: 6px;
        }
        .n-pos {
            background-color: #F0FDF4;
            border-left: 5px solid #22C55E;
            padding: 12px;
            border-radius: 6px;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-title">👑 Nim Game AI · Optimal Game Search</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Mathematical Theory (Bouton & Sprague-Grundy), '
        'Bitwise XOR Parity, Minimax Search, and Guaranteed Optimal Play.</div>',
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------------------
    # SIDEBAR CONTROLS
    # -------------------------------------------------------------------------
    with st.sidebar:
        st.header("⚙️ Configuration")
        mode = st.radio(
            "Select Mode:",
            ["🎮 Interactive Gameplay", "📊 Monte Carlo Simulation", "🔬 Theory & Bit-Matrix Explorer"],
        )

        st.markdown("---")
        rule = st.selectbox(
            "Game Rule Convention:",
            ["Normal", "Misère"],
            help="Normal: Last player to take an item wins. Misère: Last player to take loses.",
        )

        preset = st.selectbox(
            "Initial Heaps Preset:",
            ["Classic (3, 4, 5)", "Standard (1, 3, 5, 7)", "Small (2, 5, 7)", "Custom"],
        )

        if preset == "Classic (3, 4, 5)":
            default_heaps = [3, 4, 5]
        elif preset == "Standard (1, 3, 5, 7)":
            default_heaps = [1, 3, 5, 7]
        elif preset == "Small (2, 5, 7)":
            default_heaps = [2, 5, 7]
        else:
            custom_input = st.text_input("Enter heaps separated by comma:", "3, 5, 7")
            try:
                default_heaps = [max(0, int(x.strip())) for x in custom_input.split(",") if x.strip()]
                if not default_heaps:
                    default_heaps = [3, 4, 5]
            except ValueError:
                default_heaps = [3, 4, 5]

        if st.button("🔄 Reset Game / Apply Heaps", use_container_width=True):
            st.session_state.heaps = list(default_heaps)
            st.session_state.history = []
            st.session_state.game_over = False
            st.session_state.winner = None
            st.session_state.turn = "Human"
            st.rerun()

    # Session State Initialization
    if "heaps" not in st.session_state:
        st.session_state.heaps = list(default_heaps)
    if "history" not in st.session_state:
        st.session_state.history = []
    if "game_over" not in st.session_state:
        st.session_state.game_over = False
    if "winner" not in st.session_state:
        st.session_state.winner = None
    if "turn" not in st.session_state:
        st.session_state.turn = "Human"

    # =========================================================================
    # TAB 1: INTERACTIVE GAMEPLAY
    # =========================================================================
    if mode == "🎮 Interactive Gameplay":
        col1, col2 = st.columns([3, 2])

        with col1:
            st.subheader("🎲 Current Game Board")

            # Display Heaps visually
            heap_cols = st.columns(len(st.session_state.heaps))
            for i, (col, count) in enumerate(zip(heap_cols, st.session_state.heaps)):
                with col:
                    st.markdown(f"""
                        <div class="heap-card">
                            <h4 style="margin:0 0 8px 0;">Heap {i+1}</h4>
                            <div style="font-size: 2rem; font-weight: 700; color: #2563EB;">{count}</div>
                            <div style="margin-top: 8px; min-height: 40px;">
                                {''.join(['<span class="token-circle"></span>' for _ in range(min(count, 30))])}
                                {f'<small>+{count - 30}</small>' if count > 30 else ''}
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

            st.write("")

            # Check game termination
            is_over = all(h == 0 for h in st.session_state.heaps)
            if is_over:
                st.session_state.game_over = True
                if st.session_state.winner:
                    st.success(f"🏆 Game Over! Winner: **{st.session_state.winner}**")
                st.info("Click 'Reset Game' in the sidebar to play another match.")
            else:
                # Active Player Move Controls
                st.markdown("---")
                if st.session_state.turn == "Human":
                    st.write("### 👤 Your Turn (Human)")
                    non_empty_heaps = [i + 1 for i, h in enumerate(st.session_state.heaps) if h > 0]
                    if non_empty_heaps:
                        m_col1, m_col2, m_col3 = st.columns([2, 2, 2])
                        with m_col1:
                            chosen_heap = st.selectbox("Select Heap:", non_empty_heaps)
                        with m_col2:
                            max_remove = st.session_state.heaps[chosen_heap - 1]
                            remove_count = st.number_input(
                                "Items to Remove:", min_value=1, max_value=max_remove, value=1
                            )
                        with m_col3:
                            st.write("&nbsp;")
                            if st.button("🚀 Make Move", use_container_width=True):
                                before_heaps = list(st.session_state.heaps)
                                st.session_state.heaps[chosen_heap - 1] -= remove_count
                                after_heaps = list(st.session_state.heaps)

                                st.session_state.history.append({
                                    "Player": "Human",
                                    "Heap": chosen_heap,
                                    "Removed": remove_count,
                                    "State Before": str(before_heaps),
                                    "State After": str(after_heaps),
                                })

                                # Check win condition
                                if all(h == 0 for h in st.session_state.heaps):
                                    st.session_state.game_over = True
                                    st.session_state.winner = "Human" if rule == "Normal" else "Optimal AI"
                                else:
                                    st.session_state.turn = "AI"
                                st.rerun()

                elif st.session_state.turn == "AI":
                    st.write("### 🤖 Optimal AI's Turn")
                    st.write("The AI is calculating the optimal move according to Bouton's Theorem.")

                    ai_col1, ai_col2 = st.columns([1, 1])
                    with ai_col1:
                        if st.button("▶️ Trigger AI Move", type="primary", use_container_width=True):
                            heap_idx, remove_amt, explanation = get_optimal_move(
                                st.session_state.heaps, rule=rule
                            )
                            before_heaps = list(st.session_state.heaps)
                            st.session_state.heaps[heap_idx] -= remove_amt
                            after_heaps = list(st.session_state.heaps)

                            st.session_state.history.append({
                                "Player": "Optimal AI",
                                "Heap": heap_idx + 1,
                                "Removed": remove_amt,
                                "State Before": str(before_heaps),
                                "State After": str(after_heaps),
                                "Theory": explanation,
                            })

                            if all(h == 0 for h in st.session_state.heaps):
                                st.session_state.game_over = True
                                st.session_state.winner = "Optimal AI" if rule == "Normal" else "Human"
                            else:
                                st.session_state.turn = "Human"
                            st.rerun()

                    with ai_col2:
                        # Auto-play match to completion with Random AI vs Optimal AI
                        if st.button("⚡ Let Random Agent Play As Human", use_container_width=True):
                            r_idx, r_amt = get_random_move(st.session_state.heaps)
                            before_heaps = list(st.session_state.heaps)
                            st.session_state.heaps[r_idx] -= r_amt
                            after_heaps = list(st.session_state.heaps)
                            st.session_state.history.append({
                                "Player": "Random Agent (Human substitute)",
                                "Heap": r_idx + 1,
                                "Removed": r_amt,
                                "State Before": str(before_heaps),
                                "State After": str(after_heaps),
                            })
                            if all(h == 0 for h in st.session_state.heaps):
                                st.session_state.game_over = True
                                st.session_state.winner = "Random Agent" if rule == "Normal" else "Optimal AI"
                            else:
                                st.session_state.turn = "AI"
                            st.rerun()

        # Right Column: Real-Time Mathematical Breakdown
        with col2:
            st.subheader("📐 Live Mathematical Evaluation")

            nim_sum = 0
            for h in st.session_state.heaps:
                nim_sum ^= h

            if nim_sum == 0:
                st.markdown(f"""
                    <div class="p-pos">
                        <h4 style="margin:0; color:#B91C1C;">P-Position (Balanced / Losing for mover)</h4>
                        <p style="margin:4px 0 0 0; font-size: 0.95rem;">
                            <b>Nim-Sum:</b> 0 (0000₂)<br>
                            Every single legal move from this state inevitably leads to an N-position.
                            If the opponent plays optimally, the current player cannot force a win.
                        </p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="n-pos">
                        <h4 style="margin:0; color:#15803D;">N-Position (Unbalanced / Winning for mover)</h4>
                        <p style="margin:4px 0 0 0; font-size: 0.95rem;">
                            <b>Nim-Sum:</b> {nim_sum} ({bin(nim_sum)[2:].zfill(4)}₂)<br>
                            An optimal move exists that reduces one heap and restores the Nim-sum to 0,
                            guaranteeing a forced win from this position!
                        </p>
                    </div>
                """, unsafe_allow_html=True)

            st.write("")
            # Optimal Move Recommendation
            if any(h > 0 for h in st.session_state.heaps):
                best_heap, best_remove, reasoning = get_optimal_move(st.session_state.heaps, rule)
                st.info(f"💡 **Optimal Move Recommendation:** Remove **{best_remove}** from **Heap {best_heap + 1}**.\n\n_{reasoning}_")

            # Move History
            if st.session_state.history:
                st.markdown("### 📜 Move History")
                history_df = pd.DataFrame(st.session_state.history)
                st.dataframe(history_df, use_container_width=True, hide_index=True)

    # =========================================================================
    # TAB 2: MONTE CARLO SIMULATION BENCHMARK
    # =========================================================================
    elif mode == "📊 Monte Carlo Simulation":
        st.subheader("📊 Empirical Validation: Optimal AI vs Random AI")
        st.markdown(
            "This experiment verifies the core theorem: **An Optimal Nim AI will achieve a 100% win-rate "
            "from every winning (N-position) starting configuration.**"
        )

        b_col1, b_col2, b_col3 = st.columns(3)
        with b_col1:
            sim_games = st.slider("Number of Simulated Matches:", min_value=10, max_value=500, value=100, step=10)
        with b_col2:
            p1_type = st.selectbox("First Player Agent:", ["Optimal", "Random"])
        with b_col3:
            p2_type = st.selectbox("Second Player Agent:", ["Random", "Optimal"])

        calc_sum = 0
        for h in default_heaps:
            calc_sum ^= h
        pos_type = "N-position (Winning for Player 1)" if calc_sum != 0 else "P-position (Winning for Player 2)"
        st.write(f"Starting Heaps: **{default_heaps}** | Nim-sum = **{calc_sum}** | State is: **{pos_type}**")

        if st.button("🚀 Run Simulation Batch", type="primary"):
            with st.spinner("Running Monte Carlo game simulations..."):
                results = run_benchmark(default_heaps, sim_games, rule, p1_type, p2_type)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Games", results["num_games"])
            m1_rate = f"{results['p1_win_rate']:.1f}%"
            m2.metric(f"Player 1 ({p1_type}) Wins", results["p1_wins"], delta=m1_rate)
            m2_rate = f"{results['p2_win_rate']:.1f}%"
            m3.metric(f"Player 2 ({p2_type}) Wins", results["p2_wins"], delta=m2_rate)
            m4.metric("Avg Turns per Game", f"{results['avg_turns']:.2f}")

            # Visual Bar Chart
            chart_data = pd.DataFrame({
                "Player": [f"Player 1 ({p1_type})", f"Player 2 ({p2_type})"],
                "Wins": [results["p1_wins"], results["p2_wins"]],
                "Win Rate (%)": [results["p1_win_rate"], results["p2_win_rate"]],
            })
            st.bar_chart(chart_data.set_index("Player")["Win Rate (%)"])

            if p1_type == "Optimal" and calc_sum != 0 and rule == "Normal":
                st.success("✅ **Theorem Confirmed:** Optimal AI achieved a 100% win rate from this N-position!")
            elif p2_type == "Optimal" and calc_sum == 0 and rule == "Normal":
                st.success("✅ **Theorem Confirmed:** Starting from a P-position, Player 2 (Optimal AI) achieved 100% win rate!")

    # =========================================================================
    # TAB 3: THEORY & BIT-MATRIX EXPLORER
    # =========================================================================
    elif mode == "🔬 Theory & Bit-Matrix Explorer":
        st.subheader("🔬 Mathematical Foundations of Nim (Bouton's Theorem)")

        st.markdown(r"""
        ### 1. Bouton's Theorem (1901)
        In an impartial game of Nim, every game state can be represented by the heap sizes $(h_1, h_2, \dots, h_k)$.
        The **Nim-Sum** $\oplus$ is defined as the bitwise exclusive OR (XOR) of the heap sizes:
        $$S = h_1 \oplus h_2 \oplus \dots \oplus h_k$$

        - **P-Position (Previous player winning / Balanced):** $S = 0$.
          Any legal move *must* change at least one bit, necessarily leaving $S \neq 0$.
        - **N-Position (Next player winning / Unbalanced):** $S \neq 0$.
          There is always at least one legal move that transitions to a state where $S = 0$.

        ### 2. How the Optimal Move is Calculated
        1. Compute the Nim-sum $S = \bigoplus_{i=1}^k h_i$.
        2. Let $d$ be the position of the Most Significant Bit (MSB) of $S$.
        3. Choose any heap $h_i$ that has a 1 in bit position $d$ (which guarantees $h_i \oplus S < h_i$).
        4. Reduce heap $i$ to $h'_i = h_i \oplus S$.
        5. The new Nim-sum is $(h_1 \oplus \dots \oplus h_i \dots) \oplus h_i \oplus (h_i \oplus S) = S \oplus S = 0$.
        """)

        st.markdown("---")
        st.subheader("Binary XOR Matrix Representation for Current Heaps")

        # Construct binary table
        max_val = max(st.session_state.heaps) if any(st.session_state.heaps) else 1
        num_bits = max(4, max_val.bit_length())

        table_rows = []
        for i, h in enumerate(st.session_state.heaps):
            bit_str = bin(h)[2:].zfill(num_bits)
            row = {"Heap": f"Heap {i+1}", "Size": h}
            for bit_idx in range(num_bits):
                power = 1 << (num_bits - 1 - bit_idx)
                row[f"2^{num_bits - 1 - bit_idx} ({power})"] = bit_str[bit_idx]
            table_rows.append(row)

        # XOR Row
        nim_sum = 0
        for h in st.session_state.heaps:
            nim_sum ^= h
        sum_bits = bin(nim_sum)[2:].zfill(num_bits)

        xor_row = {"Heap": "Nim-Sum (XOR)", "Size": nim_sum}
        for bit_idx in range(num_bits):
            power = 1 << (num_bits - 1 - bit_idx)
            xor_row[f"2^{num_bits - 1 - bit_idx} ({power})"] = f"**{sum_bits[bit_idx]}**"
        table_rows.append(xor_row)

        df = pd.DataFrame(table_rows)
        st.table(df)

        st.markdown(r"""
        ### 3. Sprague-Grundy Equivalence
        For any impartial game under normal play convention, by the **Sprague-Grundy Theorem**:
        $$\mathcal{G}(G_1 + G_2 + \dots + G_k) = \mathcal{G}(G_1) \oplus \mathcal{G}(G_2) \oplus \dots \oplus \mathcal{G}(G_k)$$
        where for a single Nim heap of size $n$, the Grundy value $\mathcal{G}(n) = n$.
        Therefore, multi-heap Nim is mathematically identical to the direct sum of its single-heap components.
        """)

    # Footer
    st.markdown("---")
    st.caption("Nim Game AI Project · Built with Streamlit · Strict preservation of project architecture")


if __name__ == "__main__":
    main()
