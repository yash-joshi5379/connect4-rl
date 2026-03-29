import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_csv(filepath):
    """Load CSV file into DataFrame."""
    df = pd.read_csv(filepath)
    return df

def create_plots(df, output_dir=None):
    """Create 9 separate visualization windows with moving average of size 100."""
    
    plt.style.use('seaborn-v0_8-darkgrid')
    window = 100
    
    # 1. Reward
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    rolling_reward = df['reward'].rolling(window=window).mean()
    ax1.plot(df['episode'], df['reward'], alpha=0.3, color='#2E86AB', label='Raw', linewidth=0.8)
    ax1.plot(df['episode'], rolling_reward, color='#2E86AB', linewidth=2.5, label=f'Moving avg (window={window})')
    ax1.set_xlabel('Episode', fontsize=11)
    ax1.set_ylabel('Reward', fontsize=11)
    ax1.set_title('Reward over Time', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    if output_dir:
        fig1.savefig(Path(output_dir) / '1_reward.png', dpi=300, bbox_inches='tight')
    
    # 2. Loss
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    rolling_loss = df['loss'].rolling(window=window).mean()
    ax2.plot(df['episode'], df['loss'], alpha=0.3, color='#A23B72', label='Raw', linewidth=0.8)
    ax2.plot(df['episode'], rolling_loss, color='#A23B72', linewidth=2.5, label=f'Moving avg (window={window})')
    ax2.set_xlabel('Episode', fontsize=11)
    ax2.set_ylabel('Loss', fontsize=11)
    ax2.set_title('Loss over Time', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    if output_dir:
        fig2.savefig(Path(output_dir) / '2_loss.png', dpi=300, bbox_inches='tight')
    
    # 3. Epsilon
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    rolling_epsilon = df['epsilon'].rolling(window=window).mean()
    ax3.plot(df['episode'], df['epsilon'], alpha=1, color='#F18F01', label='Raw', linewidth=2.5)
    # ax3.plot(df['episode'], rolling_epsilon, color='#F18F01', linewidth=2.5, label=f'Moving avg (window={window})')
    ax3.set_xlabel('Episode', fontsize=11)
    ax3.set_ylabel('Epsilon', fontsize=11)
    ax3.set_title('Exploration Rate (Epsilon) over Time', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    fig3.tight_layout()
    if output_dir:
        fig3.savefig(Path(output_dir) / '3_epsilon.png', dpi=300, bbox_inches='tight')
    
    # 4. Buffer
    fig4, ax4 = plt.subplots(figsize=(12, 6))
    rolling_buffer = df['buffer'].rolling(window=window).mean()
    ax4.plot(df['episode'], df['buffer'], alpha=1, color='#C73E1D', label='Raw', linewidth=2.5)
    # ax4.plot(df['episode'], rolling_buffer, color='#C73E1D', linewidth=2.5, label=f'Moving avg (window={window})')
    ax4.set_xlabel('Episode', fontsize=11)
    ax4.set_ylabel('Buffer Size', fontsize=11)
    ax4.set_title('Replay Buffer Size over Time', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    fig4.tight_layout()
    if output_dir:
        fig4.savefig(Path(output_dir) / '4_buffer.png', dpi=300, bbox_inches='tight')
    
    # 5. Moves
    fig5, ax5 = plt.subplots(figsize=(12, 6))
    rolling_moves = df['moves'].rolling(window=window).mean()
    ax5.plot(df['episode'], df['moves'], alpha=0.3, color='#6A994E', label='Raw', linewidth=0.8)
    ax5.plot(df['episode'], rolling_moves, color='#6A994E', linewidth=2.5, label=f'Moving avg (window={window})')
    ax5.set_xlabel('Episode', fontsize=11)
    ax5.set_ylabel('Moves', fontsize=11)
    ax5.set_title('Moves per Episode', fontsize=13, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)
    fig5.tight_layout()
    if output_dir:
        fig5.savefig(Path(output_dir) / '5_moves.png', dpi=300, bbox_inches='tight')
    
    # 6. Outcome (mapped: -1=Loss, 0=Draw, 1=Win)
    fig6, ax6 = plt.subplots(figsize=(12, 6))
    outcome_map = {'L': -1, 'D': 0, 'W': 1}
    outcome_values = df['outcome'].map(outcome_map)
    rolling_outcome = outcome_values.rolling(window=window).mean()
    ax6.plot(df['episode'], outcome_values, alpha=0.3, color='#6A4C93', label='Raw', linewidth=0.8)
    ax6.plot(df['episode'], rolling_outcome, color='#6A4C93', linewidth=2.5, label=f'Moving avg (window={window})')
    ax6.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax6.set_xlabel('Episode', fontsize=11)
    ax6.set_ylabel('Outcome', fontsize=11)
    ax6.set_title('Outcome over Time (-1=Loss, 0=Draw, 1=Win)', fontsize=13, fontweight='bold')
    ax6.set_ylim(-1.1, 1.1)
    ax6.set_yticks([-1, 0, 1])
    ax6.set_yticklabels(['Loss', 'Draw', 'Win'])
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3)
    fig6.tight_layout()
    if output_dir:
        fig6.savefig(Path(output_dir) / '6_outcome.png', dpi=300, bbox_inches='tight')
 
    
    # 7. Reward vs Loss scatter
    fig7, ax7 = plt.subplots(figsize=(10, 8))
    scatter = ax7.scatter(df['moves'], df['reward'], c=df['episode'], cmap='viridis', s=40, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax7.set_xlabel('Moves', fontsize=11)
    ax7.set_ylabel('Reward', fontsize=11)
    ax7.set_title('Reward vs Moves (colored by episode)', fontsize=13, fontweight='bold')
    cbar = plt.colorbar(scatter, ax=ax7, label='Episode')
    ax7.grid(True, alpha=0.3)
    fig7.tight_layout()
    if output_dir:
        fig7.savefig(Path(output_dir) / '7_reward_vs_loss.png', dpi=300, bbox_inches='tight')
    
    # 8. Correlation heatmap
    # fig8, ax8 = plt.subplots(figsize=(10, 8))
    # numeric_cols = df[['reward', 'loss', 'epsilon', 'buffer', 'moves']].corr()
    # im = ax8.imshow(numeric_cols, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    # ax8.set_xticks(range(len(numeric_cols.columns)))
    # ax8.set_yticks(range(len(numeric_cols.columns)))
    # ax8.set_xticklabels(['Reward', 'Loss', 'Epsilon', 'Buffer', 'Moves'], rotation=45, ha='right', fontsize=10)
    # ax8.set_yticklabels(['Reward', 'Loss', 'Epsilon', 'Buffer', 'Moves'], fontsize=10)
    # ax8.set_title('Metric Correlations', fontsize=13, fontweight='bold')
    # cbar = plt.colorbar(im, ax=ax8)
    # for i in range(len(numeric_cols)):
    #     for j in range(len(numeric_cols)):
    #         ax8.text(j, i, f'{numeric_cols.iloc[i, j]:.2f}', ha="center", va="center", color="black", fontsize=10, fontweight='bold')
    # fig8.tight_layout()
    # if output_dir:
    #     fig8.savefig(Path(output_dir) / '8_correlations.png', dpi=300, bbox_inches='tight')
    
    # 9. All metrics normalized on same plot
    fig9, ax9 = plt.subplots(figsize=(14, 7))
    normalized_reward = (df['reward'] - df['reward'].min()) / (df['reward'].max() - df['reward'].min())
    normalized_loss = (df['loss'] - df['loss'].min()) / (df['loss'].max() - df['loss'].min())
    normalized_epsilon = (df['epsilon'] - df['epsilon'].min()) / (df['epsilon'].max() - df['epsilon'].min())
    normalized_buffer = (df['buffer'] - df['buffer'].min()) / (df['buffer'].max() - df['buffer'].min())
    normalized_moves = (df['moves'] - df['moves'].min()) / (df['moves'].max() - df['moves'].min())
    
    ax9.plot(df['episode'], normalized_reward.rolling(window=window).mean(), color='#2E86AB', linewidth=2, label='Reward', alpha=0.8)
    ax9.plot(df['episode'], normalized_loss.rolling(window=window).mean(), color='#A23B72', linewidth=2, label='Loss', alpha=0.8)
    # ax9.plot(df['episode'], normalized_epsilon.rolling(window=window).mean(), color='#F18F01', linewidth=2, label='Epsilon', alpha=0.8)
    # ax9.plot(df['episode'], normalized_buffer.rolling(window=window).mean(), color='#C73E1D', linewidth=2, label='Buffer', alpha=0.8)
    ax9.plot(df['episode'], normalized_moves.rolling(window=window).mean(), color='#6A994E', linewidth=2, label='Moves', alpha=0.8)
    
    ax9.set_xlabel('Episode', fontsize=11)
    ax9.set_ylabel('Normalized Value', fontsize=11)
    ax9.set_title(f'All Metrics Normalized (Moving Average window={window})', fontsize=13, fontweight='bold')
    ax9.legend(fontsize=11, loc='best')
    ax9.grid(True, alpha=0.3)
    ax9.set_ylim(-0.05, 1.05)
    fig9.tight_layout()
    if output_dir:
        fig9.savefig(Path(output_dir) / '9_all_metrics_normalized.png', dpi=300, bbox_inches='tight')
    
    print("All 9 plots created and displayed!")
    if output_dir:
        print(f"Plots saved to {output_dir}/")
    
    plt.show()

def print_summary(df):
    """Print summary statistics."""
    print("\n" + "="*60)
    print("TRAINING METRICS SUMMARY")
    print("="*60)
    print(f"Total episodes: {len(df)}")
    print(f"Win rate: {(df['outcome'] == 'W').sum() / len(df) * 100:.1f}%")
    print(f"\n{'Metric':<15} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
    print("-"*60)
    
    for col in ['reward', 'loss', 'epsilon', 'buffer', 'moves']:
        print(f"{col:<15} {df[col].mean():<12.4f} {df[col].std():<12.4f} {df[col].min():<12.4f} {df[col].max():<12.4f}")
    
    print("="*60 + "\n")

if __name__ == "__main__":
    # Change this to your CSV file path
    csv_file = r"history\third\training.csv"
    
    try:
        df = load_csv(csv_file)
        print_summary(df)
        create_plots(df, output_dir=".")
    except FileNotFoundError:
        print(f"Error: {csv_file} not found. Please provide the correct path.")