#!/usr/bin/env python3
"""
分析 eepose log 文件，统计每次推理的 10 步中各步的变化量分布。

用法:
    python analyze_eepose_log.py <log_file_path>
    python analyze_eepose_log.py logs/eepose_log_20260225_145440.txt
    python analyze_eepose_log.py logs/eepose_log_20260225_145440.txt --top 3 --plot

功能:
    1. 解析每次 Inference 的 10 步 eepose 和 delta
    2. Step 0 vs obs（即 Action 0 的 delta，是相对于当前观测的变化）
    3. Step 1-9 是相邻步之间的变化
    4. 统计每个 step 位置的平均/最大/最小 delta
    5. 找出每次推理中变化最大和最小的步
    6. 可选：生成可视化图表
"""

import re
import sys
import argparse
import numpy as np
from collections import defaultdict


def parse_log(log_path):
    """解析 eepose log 文件
    
    Returns:
        inferences: list of dict, 每个 dict 包含一次推理的所有信息
            {
                'inference_id': int,
                'steps': [
                    {
                        'action_idx': int,
                        'left_eepose': np.array(7),   # x,y,z,rx,ry,rz,gripper
                        'right_eepose': np.array(7),
                        'delta_left_pos': float,
                        'delta_left_euler': float,
                        'delta_left_gripper': float,
                        'delta_right_pos': float,
                        'delta_right_euler': float,
                        'delta_right_gripper': float,
                    },
                    ...
                ]
            }
    """
    with open(log_path, 'r') as f:
        content = f.read()
    
    # 提取元信息
    task_match = re.search(r'Task: (.+)', content)
    model_match = re.search(r'Model: (.+)', content)
    checkpoint_match = re.search(r'Checkpoint: (.+)', content)
    
    meta = {
        'task': task_match.group(1).strip() if task_match else 'unknown',
        'model': model_match.group(1).strip() if model_match else 'unknown',
        'checkpoint': checkpoint_match.group(1).strip() if checkpoint_match else 'unknown',
    }
    
    # 按 FRAME 分割
    frame_pattern = re.compile(
        r'FRAME (\d+)/\d+\n={80}\n(.*?)(?=\n={80}\nFRAME|\Z)',
        re.DOTALL
    )
    
    # 解析每个 step
    step_pattern = re.compile(
        r'Inference #(\d+), Step (\d+)/(\d+) \(Action (\d+)\):\n'
        r'  Left:  \[([^\]]+)\]\n'
        r'  Right: \[([^\]]+)\]\n'
        r'  Delta Left:  pos=([\d.]+) m, euler=([\d.]+) rad, gripper=([-\d.]+)\n'
        r'  Delta Right: pos=([\d.]+) m, euler=([\d.]+) rad, gripper=([-\d.]+)'
    )
    
    inferences = []
    
    for frame_match in frame_pattern.finditer(content):
        frame_id = int(frame_match.group(1))
        frame_text = frame_match.group(2)
        
        steps = []
        for step_match in step_pattern.finditer(frame_text):
            inference_id = int(step_match.group(1))
            step_num = int(step_match.group(2))
            total_steps = int(step_match.group(3))
            action_idx = int(step_match.group(4))
            
            left_eepose = np.array([float(x) for x in step_match.group(5).split(',')])
            right_eepose = np.array([float(x) for x in step_match.group(6).split(',')])
            
            steps.append({
                'step_num': step_num,
                'action_idx': action_idx,
                'left_eepose': left_eepose,
                'right_eepose': right_eepose,
                'delta_left_pos': float(step_match.group(7)),
                'delta_left_euler': float(step_match.group(8)),
                'delta_left_gripper': float(step_match.group(9)),
                'delta_right_pos': float(step_match.group(10)),
                'delta_right_euler': float(step_match.group(11)),
                'delta_right_gripper': float(step_match.group(12)),
            })
        
        if steps:
            inferences.append({
                'inference_id': frame_id,
                'steps': steps,
            })
    
    return meta, inferences


def compute_statistics(inferences):
    """计算每个 step 位置的统计量
    
    Returns:
        stats: dict, key=step_num (1-10), value=统计信息
    """
    # 按 step 位置聚合
    step_data = defaultdict(lambda: {
        'left_pos': [], 'left_euler': [], 'left_gripper': [],
        'right_pos': [], 'right_euler': [], 'right_gripper': [],
        'left_total': [], 'right_total': [], 'both_total': [],
    })
    
    for inf in inferences:
        for s in inf['steps']:
            sn = s['action_idx']  # 用 action_idx (0-9)
            step_data[sn]['left_pos'].append(s['delta_left_pos'])
            step_data[sn]['left_euler'].append(s['delta_left_euler'])
            step_data[sn]['left_gripper'].append(abs(s['delta_left_gripper']))
            step_data[sn]['right_pos'].append(s['delta_right_pos'])
            step_data[sn]['right_euler'].append(s['delta_right_euler'])
            step_data[sn]['right_gripper'].append(abs(s['delta_right_gripper']))
            # 总变化量 = pos + euler（加权组合）
            left_total = s['delta_left_pos'] + s['delta_left_euler']
            right_total = s['delta_right_pos'] + s['delta_right_euler']
            step_data[sn]['left_total'].append(left_total)
            step_data[sn]['right_total'].append(right_total)
            step_data[sn]['both_total'].append(left_total + right_total)
    
    stats = {}
    for sn in sorted(step_data.keys()):
        d = step_data[sn]
        stats[sn] = {}
        for key in d:
            arr = np.array(d[key])
            stats[sn][key] = {
                'mean': np.mean(arr),
                'std': np.std(arr),
                'min': np.min(arr),
                'max': np.max(arr),
                'median': np.median(arr),
            }
    
    return stats, step_data


def print_step_overview(stats, step_data):
    """打印每个 step 位置的概览"""
    print("\n" + "=" * 100)
    print("📊 每个 Action Step 的平均变化量 (跨所有 Inference 统计)")
    print("=" * 100)
    print(f"{'Action':>7} | {'Left Δpos':>12} {'Left Δeuler':>12} {'Right Δpos':>12} {'Right Δeuler':>12} | {'Total Δ':>12} | {'备注'}")
    print("-" * 100)
    
    for sn in sorted(stats.keys()):
        s = stats[sn]
        note = "← vs OBS (最大)" if sn == 0 else ""
        
        # 标记最大变化的 step
        total_mean = s['both_total']['mean']
        
        print(f"  [{sn:>2}]  | "
              f"{s['left_pos']['mean']:>10.6f}m "
              f"{s['left_euler']['mean']:>10.6f}r "
              f"{s['right_pos']['mean']:>10.6f}m "
              f"{s['right_euler']['mean']:>10.6f}r | "
              f"{total_mean:>10.6f} | "
              f"{note}")
    
    # 找出变化最大和最小的 step
    total_means = {sn: stats[sn]['both_total']['mean'] for sn in stats}
    max_step = max(total_means, key=total_means.get)
    
    # 排除 step 0 后找最大和最小
    non_zero_means = {sn: v for sn, v in total_means.items() if sn > 0}
    if non_zero_means:
        max_step_nz = max(non_zero_means, key=non_zero_means.get)
        min_step_nz = min(non_zero_means, key=non_zero_means.get)
        print(f"\n  ⚡ Step 0 (vs OBS) 平均总变化: {total_means[0]:.6f}")
        print(f"  ⚡ Step 1-9 中变化最大: Action [{max_step_nz}] = {non_zero_means[max_step_nz]:.6f}")
        print(f"  ⚡ Step 1-9 中变化最小: Action [{min_step_nz}] = {non_zero_means[min_step_nz]:.6f}")
        print(f"  📐 Step 0 / Step 1-9 均值比: {total_means[0] / np.mean(list(non_zero_means.values())):.2f}x")


def print_per_inference_detail(inferences, top_n=3):
    """打印每次推理的详细分析"""
    print("\n" + "=" * 100)
    print(f"📋 每次 Inference 的 Step 变化量排名 (显示变化最大/最小的 top-{top_n})")
    print("=" * 100)
    
    for inf in inferences:
        inf_id = inf['inference_id']
        steps = inf['steps']
        
        # 计算每步的总变化量
        step_totals = []
        for s in steps:
            total = (s['delta_left_pos'] + s['delta_left_euler'] + 
                     s['delta_right_pos'] + s['delta_right_euler'])
            step_totals.append((s['action_idx'], total, s))
        
        # 按总变化量排序
        sorted_steps = sorted(step_totals, key=lambda x: x[1], reverse=True)
        
        print(f"\n  Inference #{inf_id}:")
        print(f"    🔺 变化最大的 {top_n} 步:")
        for rank, (action_idx, total, s) in enumerate(sorted_steps[:top_n], 1):
            label = "(vs OBS)" if action_idx == 0 else ""
            print(f"      {rank}. Action [{action_idx}] {label}: "
                  f"L_pos={s['delta_left_pos']:.6f}, L_euler={s['delta_left_euler']:.6f}, "
                  f"R_pos={s['delta_right_pos']:.6f}, R_euler={s['delta_right_euler']:.6f} "
                  f"=> total={total:.6f}")
        
        print(f"    🔻 变化最小的 {top_n} 步:")
        for rank, (action_idx, total, s) in enumerate(sorted_steps[-top_n:], 1):
            print(f"      {rank}. Action [{action_idx}]: "
                  f"L_pos={s['delta_left_pos']:.6f}, L_euler={s['delta_left_euler']:.6f}, "
                  f"R_pos={s['delta_right_pos']:.6f}, R_euler={s['delta_right_euler']:.6f} "
                  f"=> total={total:.6f}")


def print_consistency_analysis(inferences):
    """分析各步变化量的一致性（跨inference的方差）"""
    print("\n" + "=" * 100)
    print("📈 Step 变化量一致性分析 (std/mean，越小越稳定)")
    print("=" * 100)
    
    step_totals = defaultdict(list)
    for inf in inferences:
        for s in inf['steps']:
            total = (s['delta_left_pos'] + s['delta_left_euler'] + 
                     s['delta_right_pos'] + s['delta_right_euler'])
            step_totals[s['action_idx']].append(total)
    
    print(f"{'Action':>7} | {'Mean':>10} | {'Std':>10} | {'CV(std/mean)':>12} | {'稳定性'}")
    print("-" * 70)
    
    for sn in sorted(step_totals.keys()):
        arr = np.array(step_totals[sn])
        mean = np.mean(arr)
        std = np.std(arr)
        cv = std / mean if mean > 0 else 0
        
        if cv < 0.3:
            stability = "✅ 稳定"
        elif cv < 0.6:
            stability = "⚠️ 一般"
        else:
            stability = "❌ 不稳定"
        
        print(f"  [{sn:>2}]  | {mean:>10.6f} | {std:>10.6f} | {cv:>10.4f}   | {stability}")


def print_euler_analysis(inferences):
    """分析欧拉角（朝向）变化 —— 特别关注"""
    print("\n" + "=" * 100)
    print("🧭 朝向 (Euler) 变化量分析")
    print("=" * 100)
    
    step_euler = defaultdict(lambda: {'left': [], 'right': []})
    for inf in inferences:
        for s in inf['steps']:
            step_euler[s['action_idx']]['left'].append(s['delta_left_euler'])
            step_euler[s['action_idx']]['right'].append(s['delta_right_euler'])
    
    print(f"{'Action':>7} | {'Left Euler Δ':>12} {'(std)':>8} | {'Right Euler Δ':>13} {'(std)':>8} | {'备注'}")
    print("-" * 80)
    
    for sn in sorted(step_euler.keys()):
        l_arr = np.array(step_euler[sn]['left'])
        r_arr = np.array(step_euler[sn]['right'])
        note = "← vs OBS" if sn == 0 else ""
        
        print(f"  [{sn:>2}]  | {np.mean(l_arr):>10.6f}r ({np.std(l_arr):>.4f}) | "
              f"{np.mean(r_arr):>11.6f}r ({np.std(r_arr):>.4f}) | {note}")


def print_position_trajectory(inferences):
    """打印位置轨迹概览 —— 展示从 obs 到每步的累积位移"""
    print("\n" + "=" * 100)
    print("🗺️  位置累积变化轨迹 (各 Inference 的平均)")
    print("=" * 100)
    
    n_steps = max(len(inf['steps']) for inf in inferences)
    
    # 收集每个 inference 各步的绝对 eepose
    left_pos_by_step = defaultdict(list)   # step -> list of [x,y,z]
    right_pos_by_step = defaultdict(list)
    
    for inf in inferences:
        for s in inf['steps']:
            left_pos_by_step[s['action_idx']].append(s['left_eepose'][:3])
            right_pos_by_step[s['action_idx']].append(s['right_eepose'][:3])
    
    # 计算平均轨迹
    print(f"\n  {'Action':>7} | {'Left pos (mean)':>36} | {'Right pos (mean)':>36}")
    print("  " + "-" * 85)
    
    for sn in sorted(left_pos_by_step.keys()):
        l_mean = np.mean(left_pos_by_step[sn], axis=0)
        r_mean = np.mean(right_pos_by_step[sn], axis=0)
        print(f"  [{sn:>2}]   | [{l_mean[0]:>8.4f}, {l_mean[1]:>8.4f}, {l_mean[2]:>8.4f}] | "
              f"[{r_mean[0]:>8.4f}, {r_mean[1]:>8.4f}, {r_mean[2]:>8.4f}]")


def print_heatmap_ascii(stats):
    """用 ASCII 热力图展示各步的变化量"""
    print("\n" + "=" * 100)
    print("🔥 变化量热力图 (ASCII)")
    print("=" * 100)
    
    metrics = [
        ('left_pos', 'L pos'),
        ('left_euler', 'L euler'),
        ('right_pos', 'R pos'),
        ('right_euler', 'R euler'),
    ]
    
    steps = sorted(stats.keys())
    
    # 表头
    header = f"{'Metric':>10} |"
    for sn in steps:
        header += f" [{sn:>2}] "
    print(header)
    print("-" * (12 + 7 * len(steps)))
    
    blocks = " ░▒▓█"
    
    for metric_key, metric_label in metrics:
        values = [stats[sn][metric_key]['mean'] for sn in steps]
        vmin, vmax = min(values), max(values)
        vrange = vmax - vmin if vmax > vmin else 1.0
        
        row = f"{metric_label:>10} |"
        for v in values:
            # 归一化到 0-4
            level = int((v - vmin) / vrange * 4)
            level = max(0, min(4, level))
            row += f"  {blocks[level]}   "
        
        # 附上数值范围
        row += f"  [{vmin:.4f} ~ {vmax:.4f}]"
        print(row)
    
    print(f"\n  图例: ' '=最小  ░=较小  ▒=中等  ▓=较大  █=最大")


def try_plot(stats, step_data, output_path=None):
    """尝试生成可视化图表"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n[Warning] matplotlib 未安装，跳过图表生成。安装: pip install matplotlib")
        return
    
    steps = sorted(stats.keys())
    n_steps = len(steps)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('EEPose Delta Analysis per Action Step', fontsize=14, fontweight='bold')
    
    metrics = [
        ('left_pos', 'Left Δ Position (m)', axes[0, 0]),
        ('left_euler', 'Left Δ Euler (rad)', axes[0, 1]),
        ('right_pos', 'Right Δ Position (m)', axes[1, 0]),
        ('right_euler', 'Right Δ Euler (rad)', axes[1, 1]),
    ]
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, n_steps))
    
    for metric_key, title, ax in metrics:
        means = [stats[sn][metric_key]['mean'] for sn in steps]
        stds = [stats[sn][metric_key]['std'] for sn in steps]
        
        bars = ax.bar(steps, means, yerr=stds, capsize=3, 
                      color=colors, edgecolor='black', linewidth=0.5, alpha=0.8)
        
        # 标记 step 0
        bars[0].set_edgecolor('red')
        bars[0].set_linewidth(2)
        
        ax.set_xlabel('Action Index')
        ax.set_ylabel('Delta')
        ax.set_title(title)
        ax.set_xticks(steps)
        ax.grid(axis='y', alpha=0.3)
        
        # 在 bar 上标注数值
        for i, (sn, m) in enumerate(zip(steps, means)):
            ax.text(sn, m + stds[i] * 0.1, f'{m:.4f}', ha='center', va='bottom', fontsize=7)
    
    plt.tight_layout()
    
    if output_path is None:
        output_path = 'eepose_analysis.png'
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 图表已保存到: {output_path}")
    plt.close()
    
    # 额外：生成综合对比图
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    
    left_total = [stats[sn]['left_total']['mean'] for sn in steps]
    right_total = [stats[sn]['right_total']['mean'] for sn in steps]
    both_total = [stats[sn]['both_total']['mean'] for sn in steps]
    
    x = np.array(steps)
    width = 0.25
    ax2.bar(x - width, left_total, width, label='Left Total Δ', color='steelblue', alpha=0.8)
    ax2.bar(x, right_total, width, label='Right Total Δ', color='coral', alpha=0.8)
    ax2.bar(x + width, both_total, width, label='Both Total Δ', color='mediumpurple', alpha=0.8)
    
    ax2.set_xlabel('Action Index')
    ax2.set_ylabel('Total Delta (pos + euler)')
    ax2.set_title('Total Delta per Action Step (Left / Right / Both)')
    ax2.set_xticks(steps)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    total_output = output_path.replace('.png', '_total.png')
    plt.savefig(total_output, dpi=150, bbox_inches='tight')
    print(f"📊 综合图已保存到: {total_output}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="分析 eepose log 文件")
    parser.add_argument("log_file", type=str, help="Log 文件路径")
    parser.add_argument("--top", type=int, default=3, help="每次推理显示变化最大/最小的 top N 步")
    parser.add_argument("--plot", action="store_true", help="生成可视化图表 (需要 matplotlib)")
    parser.add_argument("--output", type=str, default=None, help="图表输出路径 (默认: 与 log 同目录)")
    parser.add_argument("--no-detail", action="store_true", help="不显示每次推理的详细分析")
    args = parser.parse_args()
    
    print(f"📂 分析日志: {args.log_file}")
    
    # 解析
    meta, inferences = parse_log(args.log_file)
    
    if not inferences:
        print("❌ 未找到任何推理数据！请检查日志格式。")
        return
    
    # 元信息
    print(f"\n📌 任务: {meta['task']}")
    print(f"📌 模型: {meta['model']}")
    print(f"📌 Checkpoint: {meta['checkpoint']}")
    print(f"📌 总推理次数: {len(inferences)}")
    print(f"📌 每次推理步数: {len(inferences[0]['steps'])}")
    
    # 统计
    stats, step_data = compute_statistics(inferences)
    
    # 输出分析结果
    print_step_overview(stats, step_data)
    print_heatmap_ascii(stats)
    print_euler_analysis(inferences)
    print_consistency_analysis(inferences)
    print_position_trajectory(inferences)
    
    if not args.no_detail:
        print_per_inference_detail(inferences, top_n=args.top)
    
    # 生成图表
    if args.plot:
        if args.output:
            output_path = args.output
        else:
            import os
            log_dir = os.path.dirname(args.log_file) or '.'
            log_name = os.path.splitext(os.path.basename(args.log_file))[0]
            output_path = os.path.join(log_dir, f"{log_name}_analysis.png")
        try_plot(stats, step_data, output_path)
    
    # 简要结论
    print("\n" + "=" * 100)
    print("📝 结论摘要")
    print("=" * 100)
    
    total_means = {sn: stats[sn]['both_total']['mean'] for sn in stats}
    step0_total = total_means.get(0, 0)
    non_zero = {k: v for k, v in total_means.items() if k > 0}
    
    if non_zero:
        avg_nz = np.mean(list(non_zero.values()))
        max_step = max(non_zero, key=non_zero.get)
        min_step = min(non_zero, key=non_zero.get)
        
        print(f"  • Action [0] (vs OBS) 平均总变化 = {step0_total:.6f}，"
              f"是后续步 ({avg_nz:.6f}) 的 {step0_total / avg_nz:.1f}x")
        print(f"  • Action [1-9] 中变化最大的是 [{max_step}] = {non_zero[max_step]:.6f}")
        print(f"  • Action [1-9] 中变化最小的是 [{min_step}] = {non_zero[min_step]:.6f}")
        
        # 建议 ACTION_INDEX
        # 找到变化趋于平稳的起始点
        sorted_steps = sorted(non_zero.keys())
        stable_start = sorted_steps[0]
        for i in range(1, len(sorted_steps)):
            curr = sorted_steps[i]
            prev = sorted_steps[i - 1]
            if non_zero[curr] <= avg_nz * 0.8:
                stable_start = curr
                break
        
        print(f"\n  💡 建议:")
        print(f"     - Action [0] 变化量显著大于其他步 (主要是从当前状态跳到预测)")
        if step0_total > avg_nz * 3:
            print(f"     - 考虑跳过 Action [0]，从 ACTION_INDEX=1 开始执行")
        else:
            print(f"     - Action [0] 变化量适中，可以从 ACTION_INDEX=0 开始执行")
        
        # 检查后半段是否变化很小（收敛）
        second_half = [non_zero[s] for s in sorted_steps if s >= len(sorted_steps) // 2]
        first_half = [non_zero[s] for s in sorted_steps if s < len(sorted_steps) // 2]
        if first_half and second_half:
            ratio = np.mean(second_half) / np.mean(first_half)
            if ratio < 0.5:
                print(f"     - 后半段 (Action [{len(sorted_steps)//2}-9]) 变化量为前半段的 {ratio:.1%}，可考虑减少 EXECUTE_STEPS")
    
    print("\n✅ 分析完成！")


if __name__ == "__main__":
    main()
