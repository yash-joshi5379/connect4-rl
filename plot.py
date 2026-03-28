"""
Plot AlphaZero training metrics from a training_log.txt file.
Produces 5 plots — no moving averages:
  1. Total loss        (every iter)
  2. Value loss        (every iter)
  3. Policy loss       (every iter)
  4. Replay buffer     (every iter)
  5. Last self-play game moves  (only at logged intervals)
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt

# ── set your log path here ───────────────────────────────────────────────────
# MODEL_DIR = r"SUCCESS_400_20_150"
# MODEL_DIR = r"SUCCESS_400_30_180"
MODEL_DIR = r"SUCCESS_800_40_200"
PATH = Path(MODEL_DIR) / "training_log.txt"
# ── optionally save plots to a directory (set to None to skip) ───────────────
SAVE_DIR = Path(MODEL_DIR) / "plots"
SAVE_DIR.mkdir(exist_ok=True)
# ─────────────────────────────────────────────────────────────────────────────


# ── colours from the original visualisation script ──────────────────────────
COLOR_LOSS   = '#A23B72'   # total loss
COLOR_VLOSS  = '#2E86AB'   # value loss
COLOR_PLOSS  = '#6A4C93'   # policy loss
COLOR_BUFFER = '#C73E1D'   # buffer
COLOR_MOVES  = '#6A994E'   # moves


# ── log parser ───────────────────────────────────────────────────────────────

def parse_log(log_path):
    iter_re  = re.compile(
        r'Iter\s+(\d+)\s*\|.*?loss=([\d.]+)\s*\(v=([\d.]+)\s+p=([\d.]+)\)'
        r'.*?\|\s*buf=\s*(\d+)'
    )
    moves_re = re.compile(r'Last self-play game \((\d+) moves')

    iters, loss, v_loss, p_loss, buf = [], [], [], [], []
    move_iters, moves = [], []
    current_iter = None

    with open(log_path, 'r') as f:
        for line in f:
            m = iter_re.search(line)
            if m:
                current_iter = int(m.group(1))
                iters.append(current_iter)
                loss.append(float(m.group(2)))
                v_loss.append(float(m.group(3)))
                p_loss.append(float(m.group(4)))
                buf.append(int(m.group(5)))
                continue

            m2 = moves_re.search(line)
            if m2 and current_iter is not None:
                move_iters.append(current_iter)
                moves.append(int(m2.group(1)))

    return dict(
        iters=iters, loss=loss, v_loss=v_loss, p_loss=p_loss,
        buffer=buf, move_iters=move_iters, moves=moves,
    )


# ── shared chart helper ───────────────────────────────────────────────────────

def _make_ax(title, ylabel):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, alpha=0.3)
    return fig, ax


# ── plot functions ────────────────────────────────────────────────────────────

def plot_loss(data, save_path=None):
    fig, ax = _make_ax('Total Loss over Iterations', 'Loss')
    ax.plot(data['iters'], data['loss'], color=COLOR_LOSS, linewidth=1.8, label='loss')
    ax.legend(fontsize=10)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')


def plot_value_loss(data, save_path=None):
    fig, ax = _make_ax('Value Loss over Iterations', 'Value Loss')
    ax.plot(data['iters'], data['v_loss'], color=COLOR_VLOSS, linewidth=1.8, label='value loss')
    ax.legend(fontsize=10)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')


def plot_policy_loss(data, save_path=None):
    fig, ax = _make_ax('Policy Loss over Iterations', 'Policy Loss')
    ax.plot(data['iters'], data['p_loss'], color=COLOR_PLOSS, linewidth=1.8, label='policy loss')
    ax.legend(fontsize=10)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')


def plot_buffer(data, save_path=None):
    fig, ax = _make_ax('Replay Buffer Size over Iterations', 'Buffer Size')
    ax.plot(data['iters'], data['buffer'], color=COLOR_BUFFER, linewidth=1.8, label='buffer')
    ax.legend(fontsize=10)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')


def plot_moves(data, save_path=None):
    fig, ax = _make_ax('Last Self-Play Game — Moves', 'Moves')
    ax.plot(data['move_iters'], data['moves'], color=COLOR_MOVES, linewidth=1.8, label='game moves')
    ax.legend(fontsize=10)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    data = parse_log(PATH)

    print(f"Iterations parsed : {len(data['iters'])}")
    if data['iters']:
        print(f"Iter range        : {data['iters'][0]} → {data['iters'][-1]}")
    print(f"Move snapshots    : {len(data['moves'])}")

    plt.style.use('seaborn-v0_8-darkgrid')

    # plots = [
    #     ('1_loss.png',        plot_loss),
    #     ('2_value_loss.png',  plot_value_loss),
    #     ('3_policy_loss.png', plot_policy_loss),
    #     ('4_buffer.png',      plot_buffer),
    #     ('5_moves.png',       plot_moves),
    # ]

    plots = [
        ('1_value_loss.png',  plot_value_loss),
        ('2_policy_loss.png', plot_policy_loss),
        ('3_buffer.png',      plot_buffer),
        ('4_moves.png',       plot_moves),
    ]

    for fname, fn in plots:
        save = str(Path(SAVE_DIR) / fname) if SAVE_DIR else None
        fn(data, save_path=save)

    if SAVE_DIR:
        print(f"Saved to: {SAVE_DIR}/")

    plt.show()


if __name__ == '__main__':
    main()