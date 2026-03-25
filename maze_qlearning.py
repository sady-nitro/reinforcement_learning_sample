"""
Q-Learning による 20x20 迷路問題の解法

迷路構成:
  - 20x20 グリッド (外周は壁)
  - S: スタート, G: ゴール, #: 壁, .: 通路
  - エージェントは上下左右の4方向に移動可能

Q-Learning パラメータ:
  - 学習率 α, 割引率 γ, 探索率 ε (ε-greedy)
"""

import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib

matplotlib.rcParams["font.family"] = "sans-serif"

# ============================================================
# 迷路定義 (20x20)
# ============================================================
# 0: 通路, 1: 壁, 2: スタート, 3: ゴール
MAZE = [
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,2,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1],
    [1,0,1,0,1,0,1,1,1,1,1,0,1,0,1,1,1,1,0,1],
    [1,0,1,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,1],
    [1,0,1,1,1,1,1,1,1,0,1,1,1,0,1,0,1,1,1,1],
    [1,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,1],
    [1,1,1,0,1,1,1,0,1,1,1,1,1,1,1,0,1,1,0,1],
    [1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1],
    [1,0,1,1,1,0,1,1,1,0,1,1,1,1,1,0,1,0,1,1],
    [1,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,1],
    [1,1,1,1,1,0,1,0,1,1,1,0,1,0,1,1,1,1,0,1],
    [1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,1,0,1],
    [1,0,1,1,1,1,1,1,1,1,1,0,1,1,1,1,0,1,0,1],
    [1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,1],
    [1,0,1,1,1,0,1,1,1,0,1,1,1,1,0,1,1,1,0,1],
    [1,0,1,0,0,0,1,0,1,0,0,0,0,1,0,0,0,1,0,1],
    [1,0,1,0,1,1,1,0,1,1,1,1,0,1,1,1,0,1,0,1],
    [1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1],
    [1,0,1,0,1,0,1,1,1,1,1,0,1,1,0,0,0,1,3,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
]

ROWS = len(MAZE)
COLS = len(MAZE[0])

# 行動: 上, 下, 左, 右
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ACTION_NAMES = ["↑", "↓", "←", "→"]
NUM_ACTIONS = len(ACTIONS)

# スタートとゴールの座標を取得
START = None
GOAL = None
for r in range(ROWS):
    for c in range(COLS):
        if MAZE[r][c] == 2:
            START = (r, c)
        elif MAZE[r][c] == 3:
            GOAL = (r, c)


def is_passable(r, c):
    """指定座標が通行可能かどうか"""
    return 0 <= r < ROWS and 0 <= c < COLS and MAZE[r][c] != 1


# ============================================================
# 報酬設定
# ============================================================
REWARD_GOAL = 100.0   # ゴール到達
REWARD_WALL = -1.0    # 壁への衝突 (移動失敗)
REWARD_STEP = -0.1    # 通常の1ステップ


def get_reward_and_next(state, action_idx):
    """行動を実行し、(次の状態, 報酬, 終了フラグ) を返す"""
    r, c = state
    dr, dc = ACTIONS[action_idx]
    nr, nc = r + dr, c + dc

    if not is_passable(nr, nc):
        return state, REWARD_WALL, False

    next_state = (nr, nc)
    if next_state == GOAL:
        return next_state, REWARD_GOAL, True

    return next_state, REWARD_STEP, False


# ============================================================
# Q-Learning
# ============================================================
def q_learning(
    episodes=5000,
    alpha=0.1,       # 学習率
    gamma=0.99,      # 割引率
    epsilon=1.0,     # 初期探索率
    epsilon_min=0.01,
    epsilon_decay=0.999,
    max_steps=1000,
    snapshot_intervals=None,
    trajectory_intervals=None,
):
    """Q-Learning で迷路を学習する

    Returns:
        q_table, stats dict (rewards, steps, epsilons, snapshots)
    """
    if snapshot_intervals is None:
        snapshot_intervals = [1, 50, 200, 500, 1000, 2000, 3000, 5000]
    if trajectory_intervals is None:
        trajectory_intervals = []

    # Q テーブル: (row, col) × action
    q_table = np.zeros((ROWS, COLS, NUM_ACTIONS))

    rewards_history = []
    steps_history = []
    epsilon_history = []
    snapshots = {}  # episode -> q_table snapshot
    trajectories = {}  # episode -> (軌跡リスト, その時点のq_table)

    for ep in range(1, episodes + 1):
        state = START
        total_reward = 0.0
        step_count = 0
        record_traj = ep in trajectory_intervals
        trajectory = [state] if record_traj else None

        for _ in range(max_steps):
            # ε-greedy 行動選択
            if random.random() < epsilon:
                action = random.randrange(NUM_ACTIONS)
            else:
                action = int(np.argmax(q_table[state[0], state[1]]))

            next_state, reward, done = get_reward_and_next(state, action)
            total_reward += reward
            step_count += 1

            # Q値更新
            r, c = state
            nr, nc = next_state
            best_next = np.max(q_table[nr, nc])
            q_table[r, c, action] += alpha * (
                reward + gamma * best_next - q_table[r, c, action]
            )

            state = next_state
            if record_traj:
                trajectory.append(state)
            if done:
                break

        rewards_history.append(total_reward)
        steps_history.append(step_count)
        epsilon_history.append(epsilon)

        # 軌跡保存
        if record_traj:
            trajectories[ep] = (trajectory, q_table.copy())

        # スナップショット保存
        if ep in snapshot_intervals:
            snapshots[ep] = q_table.copy()

        # ε を減衰
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # 進捗表示
        if ep % 500 == 0:
            avg = np.mean(rewards_history[-100:])
            print(f"Episode {ep:5d} | ε={epsilon:.4f} | 直近100ep平均報酬: {avg:.2f}")

    stats = {
        "rewards": rewards_history,
        "steps": steps_history,
        "epsilons": epsilon_history,
        "snapshots": snapshots,
        "trajectories": trajectories,
    }
    return q_table, stats


# ============================================================
# 学習済みポリシーで最短経路を抽出
# ============================================================
def extract_path(q_table, max_steps=400):
    """学習済み Q テーブルからグリーディにゴールまでのパスを返す"""
    state = START
    path = [state]
    visited = {state}

    for _ in range(max_steps):
        if state == GOAL:
            break
        action = int(np.argmax(q_table[state[0], state[1]]))
        next_state, _, done = get_reward_and_next(state, action)
        if next_state in visited:
            # ループ検出 — 学習不足
            break
        visited.add(next_state)
        path.append(next_state)
        state = next_state
        if done:
            break

    return path


# ============================================================
# 表示ユーティリティ
# ============================================================
CELL_MAP = {0: " . ", 1: "###", 2: " S ", 3: " G "}


def print_maze(path=None):
    """迷路を表示する。path が与えられた場合は経路を '*' で重ねる"""
    path_set = set(path) if path else set()
    lines = []
    for r in range(ROWS):
        row_str = ""
        for c in range(COLS):
            if (r, c) == START:
                row_str += " S "
            elif (r, c) == GOAL:
                row_str += " G "
            elif (r, c) in path_set:
                row_str += " * "
            else:
                row_str += CELL_MAP[MAZE[r][c]]
        lines.append(row_str)
    print("\n".join(lines))


def print_policy(q_table):
    """学習済みポリシー (各セルでの最適行動方向) を表示"""
    lines = []
    for r in range(ROWS):
        row_str = ""
        for c in range(COLS):
            if MAZE[r][c] == 1:
                row_str += " # "
            elif (r, c) == GOAL:
                row_str += " G "
            else:
                best = int(np.argmax(q_table[r, c]))
                row_str += f" {ACTION_NAMES[best]} "
        lines.append(row_str)
    print("\n".join(lines))


# ============================================================
# 可視化
# ============================================================
OUTPUT_DIR = "."


def _moving_average(data, window=100):
    """移動平均を計算"""
    cumsum = np.cumsum(data)
    cumsum[window:] = cumsum[window:] - cumsum[:-window]
    result = np.full(len(data), np.nan)
    result[window - 1 :] = cumsum[window - 1 :] / window
    return result


def plot_learning_curves(stats):
    """学習曲線を描画 (報酬・ステップ数・ε の推移)"""
    rewards = stats["rewards"]
    steps = stats["steps"]
    epsilons = stats["epsilons"]
    episodes = np.arange(1, len(rewards) + 1)
    window = 100

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("Q-Learning Training Progress", fontsize=16, fontweight="bold")

    # --- 1) エピソード報酬 ---
    ax = axes[0]
    ax.plot(episodes, rewards, alpha=0.25, color="steelblue", linewidth=0.5,
            label="Episode reward")
    avg = _moving_average(rewards, window)
    ax.plot(episodes, avg, color="darkblue", linewidth=2,
            label=f"Moving avg ({window} ep)")
    ax.set_ylabel("Total Reward")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    # --- 2) ステップ数 ---
    ax = axes[1]
    ax.plot(episodes, steps, alpha=0.25, color="coral", linewidth=0.5,
            label="Steps / episode")
    avg_steps = _moving_average(steps, window)
    ax.plot(episodes, avg_steps, color="darkred", linewidth=2,
            label=f"Moving avg ({window} ep)")
    ax.set_ylabel("Steps")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # --- 3) ε (探索率) ---
    ax = axes[2]
    ax.plot(episodes, epsilons, color="green", linewidth=2)
    ax.set_ylabel("Epsilon")
    ax.set_xlabel("Episode")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/learning_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  -> {path}")


def _draw_maze_on_ax(ax, q_table, path=None, title=""):
    """Axes 上に迷路・Q値ヒートマップ・経路を描画"""
    maze_arr = np.array(MAZE, dtype=float)

    # Q値の最大値をヒートマップとして表示 (通路セルのみ)
    q_max = np.max(q_table, axis=2)
    display = np.full((ROWS, COLS), np.nan)
    for r in range(ROWS):
        for c in range(COLS):
            if maze_arr[r][c] != 1:
                display[r][c] = q_max[r][c]

    # 壁を灰色で塗る
    wall_img = np.ones((ROWS, COLS, 4))  # RGBA
    for r in range(ROWS):
        for c in range(COLS):
            if maze_arr[r][c] == 1:
                wall_img[r, c] = [0.2, 0.2, 0.2, 1.0]  # 暗灰色
            else:
                wall_img[r, c] = [1, 1, 1, 0]  # 透明

    # Q値ヒートマップ
    vmin = np.nanmin(display) if not np.all(np.isnan(display)) else 0
    vmax = np.nanmax(display) if not np.all(np.isnan(display)) else 1
    if vmin == vmax:
        vmax = vmin + 1
    ax.imshow(display, cmap="YlOrRd", vmin=vmin, vmax=vmax,
              origin="upper", interpolation="nearest")
    ax.imshow(wall_img, origin="upper", interpolation="nearest")

    # 経路を線で描画
    if path and len(path) > 1:
        pr = [p[0] for p in path]
        pc = [p[1] for p in path]
        ax.plot(pc, pr, color="blue", linewidth=2.5, alpha=0.8, zorder=3)
        ax.plot(pc[0], pr[0], "o", color="lime", markersize=8, zorder=4)
        ax.plot(pc[-1], pr[-1], "s", color="red", markersize=8, zorder=4)

    # ポリシー矢印 (通路セルのみ)
    arrow_map = {0: (0, -0.3), 1: (0, 0.3), 2: (-0.3, 0), 3: (0.3, 0)}
    for r in range(ROWS):
        for c in range(COLS):
            if maze_arr[r][c] == 1 or (r, c) == GOAL:
                continue
            if np.all(q_table[r, c] == 0):
                continue
            best = int(np.argmax(q_table[r, c]))
            dx, dy = arrow_map[best]
            ax.annotate("", xy=(c + dx, r + dy), xytext=(c, r),
                        arrowprops=dict(arrowstyle="->", color="black",
                                        lw=1.2, alpha=0.6))

    # S / G ラベル
    ax.text(START[1], START[0], "S", ha="center", va="center",
            fontsize=9, fontweight="bold", color="green", zorder=5)
    ax.text(GOAL[1], GOAL[0], "G", ha="center", va="center",
            fontsize=9, fontweight="bold", color="red", zorder=5)

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])


def plot_policy_snapshots(stats):
    """学習途中のポリシー変化をスナップショットで表示"""
    snapshots = stats["snapshots"]
    eps_list = sorted(snapshots.keys())
    n = len(eps_list)
    if n == 0:
        return

    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    if n == 1:
        axes = np.array([axes])
    axes = np.array(axes).flatten()

    for i, ep in enumerate(eps_list):
        qt = snapshots[ep]
        path = extract_path(qt)
        reached = path and path[-1] == GOAL
        label = f"Episode {ep}" + (" (GOAL)" if reached else "")
        _draw_maze_on_ax(axes[i], qt, path if reached else None, title=label)

    # 余った Axes を非表示に
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Policy Evolution During Training\n(Heatmap = max Q-value, Arrows = greedy policy)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/policy_snapshots.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  -> {path}")


def plot_final_result(q_table, path):
    """最終結果 (迷路 + 最適経路 + Q値ヒートマップ) を1枚に描画"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    _draw_maze_on_ax(ax, q_table, path,
                     title=f"Learned Optimal Path ({len(path)-1} steps)")
    plt.tight_layout()
    fpath = f"{OUTPUT_DIR}/final_result.png"
    fig.savefig(fpath, dpi=150)
    plt.close(fig)
    print(f"  -> {fpath}")


def animate_episode(trajectory, q_table, episode_num, fps=10):
    """エピソード中のエージェントの移動をアニメーションGIFとして保存する

    Args:
        trajectory: エージェントが訪問した (row, col) のリスト
        q_table: その時点での Q テーブル
        episode_num: エピソード番号 (ファイル名に使用)
        fps: フレームレート
    """
    maze_arr = np.array(MAZE, dtype=float)

    # Q値ヒートマップ用データ
    q_max = np.max(q_table, axis=2)
    display = np.full((ROWS, COLS), np.nan)
    for r in range(ROWS):
        for c in range(COLS):
            if maze_arr[r][c] != 1:
                display[r][c] = q_max[r][c]

    # 壁用 RGBA レイヤー
    wall_img = np.ones((ROWS, COLS, 4))
    for r in range(ROWS):
        for c in range(COLS):
            if maze_arr[r][c] == 1:
                wall_img[r, c] = [0.2, 0.2, 0.2, 1.0]
            else:
                wall_img[r, c] = [1, 1, 1, 0]

    vmin = np.nanmin(display) if not np.all(np.isnan(display)) else 0
    vmax = np.nanmax(display) if not np.all(np.isnan(display)) else 1
    if vmin == vmax:
        vmax = vmin + 1

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # 背景描画 (一度だけ)
    ax.imshow(display, cmap="YlOrRd", vmin=vmin, vmax=vmax,
              origin="upper", interpolation="nearest")
    ax.imshow(wall_img, origin="upper", interpolation="nearest")
    ax.text(START[1], START[0], "S", ha="center", va="center",
            fontsize=9, fontweight="bold", color="green", zorder=5)
    ax.text(GOAL[1], GOAL[0], "G", ha="center", va="center",
            fontsize=9, fontweight="bold", color="red", zorder=5)
    ax.set_xticks([])
    ax.set_yticks([])

    reached_goal = trajectory[-1] == GOAL
    total_steps = len(trajectory) - 1
    title = ax.set_title(f"Episode {episode_num} — Step 0/{total_steps}", fontsize=12, fontweight="bold")

    # アニメーション用オブジェクト
    trail_line, = ax.plot([], [], color="blue", linewidth=2, alpha=0.5, zorder=3)
    agent_dot, = ax.plot([], [], "o", color="lime", markersize=12,
                         markeredgecolor="darkgreen", markeredgewidth=1.5, zorder=6)

    # 軌跡が長い場合はフレームを間引く (最大200フレーム)
    max_frames = 200
    if len(trajectory) > max_frames:
        indices = np.linspace(0, len(trajectory) - 1, max_frames, dtype=int)
        # 最終フレームを必ず含める
        if indices[-1] != len(trajectory) - 1:
            indices[-1] = len(trajectory) - 1
    else:
        indices = np.arange(len(trajectory))

    def update(frame_idx):
        idx = indices[frame_idx]
        # ここまでの軌跡
        trail = trajectory[: idx + 1]
        tr = [p[0] for p in trail]
        tc = [p[1] for p in trail]
        trail_line.set_data(tc, tr)

        # エージェント現在位置
        cr, cc = trajectory[idx]
        agent_dot.set_data([cc], [cr])

        step_label = idx
        suffix = ""
        if idx == len(trajectory) - 1 and reached_goal:
            suffix = " — GOAL!"
        title.set_text(f"Episode {episode_num} — Step {step_label}/{total_steps}{suffix}")

        return trail_line, agent_dot, title

    anim = FuncAnimation(fig, update, frames=len(indices), interval=1000 // fps, blit=True)
    fpath = f"{OUTPUT_DIR}/episode_{episode_num}_animation.gif"
    anim.save(fpath, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"  -> {fpath}")


def animate_episodes(stats, fps=10):
    """記録されたすべてのエピソード軌跡のアニメーションを生成"""
    trajectories = stats.get("trajectories", {})
    for ep in sorted(trajectories.keys()):
        trajectory, q_table = trajectories[ep]
        animate_episode(trajectory, q_table, ep, fps=fps)


# ============================================================
# メイン
# ============================================================
def main():
    print("=" * 60)
    print("  Q-Learning による 20×20 迷路問題")
    print("=" * 60)
    print()

    print("【迷路】")
    print_maze()
    print()

    print("【学習開始】")
    # アニメーション化するエピソード (学習初期・中期・後期)
    anim_episodes = [1, 10, 50, 200, 1000, 5000]
    q_table, stats = q_learning(
        episodes=5000,
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.999,
        max_steps=1000,
        trajectory_intervals=anim_episodes,
    )
    print()

    # 経路抽出
    path = extract_path(q_table)
    if path and path[-1] == GOAL:
        print(f"【学習結果】ゴールへの経路を発見! (ステップ数: {len(path) - 1})")
    else:
        print("【学習結果】ゴールへの経路が見つかりませんでした。エピソード数を増やしてください。")

    print()
    print("【経路付き迷路】( * = エージェントの経路 )")
    print_maze(path)

    print()
    print("【学習済みポリシー】(各セルでの最適行動方向)")
    print_policy(q_table)

    # --- 可視化 ---
    print()
    print("【グラフ出力】")
    plot_learning_curves(stats)
    plot_policy_snapshots(stats)
    plot_final_result(q_table, path)
    print()
    print("【アニメーション出力】")
    animate_episodes(stats, fps=10)
    print()
    print("可視化画像・アニメーションを出力しました。")


if __name__ == "__main__":
    main()
