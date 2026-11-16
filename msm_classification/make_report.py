#!/usr/bin/env python3
"""Create the MSM classification report from completed experiment artifacts.

Run after both classification_report.json files and Dataset/Data/msm_prepared/summary.json exist:
    python make_report.py

No data are simulated and no model is trained or evaluated by this script.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import sys
from pathlib import Path
from urllib.parse import quote

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from Dataset.Preprocessing.dataset_splits import SPLIT_NAMES, read_split_csv, split_recording_ids


CLASS_LABELS = ['正常切削', '异常切削，需要干预', '缺陷刀具下的正常切削']
AXIS_LABELS = ['0 正常切削', '1 异常切削', '2 缺陷刀具\n正常切削']
COLORS = ['#237FA6', '#BF703C']


def read_json(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f'Required completed result is missing: {path}')
    return json.loads(path.read_text(encoding='utf-8-sig'))


def h(value) -> str:
    return html.escape(str(value), quote=True)


def pct(value) -> str:
    return f'{float(value)*100:.2f}%'


def number(value) -> str:
    return f'{int(value):,}'


def elapsed(value) -> str:
    seconds = float(value)
    if seconds >= 3600:
        return f'{seconds/3600:.2f}小时'
    if seconds >= 60:
        return f'{seconds/60:.2f}分钟'
    return f'{seconds:.2f}秒'


def link(path: Path, label: str, output: Path) -> str:
    relative = Path(os.path.relpath(path.resolve(), output.resolve())).as_posix()
    return f'<a href="{h(quote(relative, safe="/"))}">{h(label)}</a>'


def table(headers, rows) -> str:
    head = ''.join(f'<th>{h(value)}</th>' for value in headers)
    body = ''.join('<tr>' + ''.join(f'<td>{h(value)}</td>' for value in row) + '</tr>' for row in rows)
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def verify_metrics(metrics: dict, name: str) -> None:
    matrix = np.asarray(metrics['confusion_matrix'])
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or np.any(matrix < 0):
        raise ValueError(f'{name}: expected a finite nonnegative 3x3 confusion matrix')
    if not np.equal(matrix, np.floor(matrix)).all():
        raise ValueError(f'{name}: confusion matrix must contain sample counts')
    per_class = sorted(metrics['per_class'], key=lambda item: item['class_id'])
    if [row['class_id'] for row in per_class] != [0, 1, 2]:
        raise ValueError(f'{name}: the three class IDs must be 0, 1, 2')
    if not np.array_equal(matrix.sum(axis=1), [row['support'] for row in per_class]):
        raise ValueError(f'{name}: per-class support differs from confusion matrix rows')
    for key in ('accuracy', 'balanced_accuracy', 'macro_f1'):
        if not np.isfinite(metrics[key]) or not 0 <= metrics[key] <= 1:
            raise ValueError(f'{name}: invalid {key}')


def canonical_split(split: dict, name: str) -> dict:
    if set(split) != set(SPLIT_NAMES):
        raise ValueError(f'{name}: expected train, val and test assignments')
    result = {}
    for part in SPLIT_NAMES:
        ids = split[part]
        if not isinstance(ids, list) or not ids or any(not isinstance(value, str) or not value for value in ids):
            raise ValueError(f'{name}: {part} must contain recording IDs')
        result[part] = sorted(ids)
    flat = [value for ids in result.values() for value in ids]
    if len(flat) != len(set(flat)):
        raise ValueError(f'{name}: duplicated or overlapping recording assignments')
    return result


def validate_comparable_models(models: list[dict]) -> None:
    first = models[0]
    for model in models[1:]:
        if model['source_split'] != first['source_split'] or model['split'] != first['split']:
            raise ValueError('The two models used different actual source-recording splits')
        if model['report']['split_window_counts'] != first['report']['split_window_counts']:
            raise ValueError('The two models were not evaluated on the same split window counts')


def load_model(name: str, directory: Path) -> dict:
    report = read_json(directory/'classification_report.json')
    history = read_json(directory/'history.json')
    if not isinstance(history, list) or not history:
        raise ValueError(f'{name}: training history is empty')
    epochs = [row['epoch'] for row in history]
    if epochs != sorted(set(epochs)) or report['best_epoch'] not in epochs:
        raise ValueError(f'{name}: inconsistent epochs or selected checkpoint')
    if report['epochs_run'] != epochs[-1]:
        raise ValueError(f'{name}: completed epoch differs from saved history')
    for row in history:
        for key in ('train_loss', 'train_ce_loss'):
            value = row.get(key)
            if value is None and row.get('resumed_anchor'):
                continue
            if value is None or not np.isfinite(value):
                raise ValueError(f'{name}: non-finite learning-curve value for {key}')
        for key in ('val_interval_macro_f1', 'val_interval_accuracy'):
            if row.get(key) is not None and not np.isfinite(row[key]):
                raise ValueError(f'{name}: non-finite learning-curve value for {key}')
    selected = next(row for row in history if row['epoch'] == report['best_epoch'])
    if selected.get('val_interval_macro_f1') is None or selected.get('val_interval_accuracy') is None:
        raise ValueError(f'{name}: best_epoch is not an actual validation evaluation point')
    for split in ('validation', 'test'):
        for unit in ('interval_metrics', 'window_metrics'):
            verify_metrics(report[split][unit], f'{name}/{split}/{unit}')
    verify_metrics(report['majority_baseline_interval_metrics'], f'{name}/majority baseline')
    config_path = directory/'training_config.json'
    config = read_json(config_path) if config_path.is_file() else {}
    split = canonical_split(read_json(directory/'data_split.json'), name)
    source_split = canonical_split(report.get('split_source_recordings', split), name)
    if source_split != split:
        raise ValueError(f'{name}: MSM source IDs must equal its recording IDs')
    snapshot = directory/'dataset_splits_used.csv'
    if snapshot.is_file():
        rows = read_split_csv(snapshot)
        if any(row['source_recording_id'] != row['recording_id'] for row in rows):
            raise ValueError(f'{name}: MSM snapshot contains invented source groups')
        if canonical_split(split_recording_ids(rows), name) != split:
            raise ValueError(f'{name}: saved CSV snapshot and data_split.json disagree')
    counts = {
        'train': report['train_class_counts_windows'],
        'val': np.asarray(report['validation']['window_metrics']['confusion_matrix']).sum(axis=1).tolist(),
        'test': np.asarray(report['test']['window_metrics']['confusion_matrix']).sum(axis=1).tolist(),
    }
    for part in SPLIT_NAMES:
        if len(split[part]) != report['split_recording_counts'][part]:
            raise ValueError(f'{name}: {part} recording count differs from data_split.json')
        if sum(counts[part]) != report['split_window_counts'][part]:
            raise ValueError(f'{name}: {part} class counts differ from this run window count')
    return {'name': name, 'directory': directory, 'report': report, 'history': history,
            'config': config, 'split': split, 'source_split': source_split, 'split_class_counts': counts}


def style_axes(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(color='#DFE6EB', linewidth=.65, axis='y')
    ax.set_axisbelow(True)


def draw_confusion_matrices(models: list[dict], output: Path) -> None:
    fig = plt.figure(figsize=(13.5, 6.7))
    grid = fig.add_gridspec(1, 3, width_ratios=[1, 1, .045], left=.08, right=.94,
                          top=.80, bottom=.24, wspace=.29)
    fig.suptitle('测试集：标注切削区间级混淆矩阵', x=.06, y=.965, ha='left', fontsize=19)
    fig.text(.06, .895, '每格显示区间数量和该真实类别内的百分比；行是真实类别，列是预测类别。', fontsize=11)
    for index, model in enumerate(models):
        metrics = model['report']['test']['interval_metrics']
        counts = np.asarray(metrics['confusion_matrix'], dtype=np.int64)
        denominator = counts.sum(axis=1, keepdims=True)
        proportions = np.divide(counts, denominator, out=np.zeros_like(counts, dtype=float), where=denominator != 0)
        ax = fig.add_subplot(grid[0, index])
        image = ax.imshow(proportions, cmap='Blues', vmin=0, vmax=1, interpolation='nearest', aspect='equal')
        for row in range(3):
            for column in range(3):
                label = f'{counts[row, column]}\n({proportions[row, column]*100:.1f}%)'
                ax.text(column, row, label, ha='center', va='center', fontsize=13,
                        color='white' if proportions[row, column] >= .55 else '#17334A')
        ax.set_xticks(range(3), AXIS_LABELS, fontsize=10)
        ax.set_yticks(range(3), AXIS_LABELS, fontsize=10)
        ax.tick_params(length=0, pad=7)
        ax.set_xlabel('预测类别', labelpad=10)
        ax.set_ylabel('真实类别', labelpad=8)
        ax.set_title(f"{model['name']}\nAccuracy {pct(metrics['accuracy'])}  |  Macro F1 {pct(metrics['macro_f1'])}",
                     fontsize=12, pad=14)
        for spine in ax.spines.values():
            spine.set_visible(False)
    color_axis = fig.add_subplot(grid[0, 2])
    colorbar = fig.colorbar(image, cax=color_axis)
    colorbar.ax.yaxis.set_major_formatter(PercentFormatter(1))
    colorbar.set_label('真实类别内比例', fontsize=10)
    fig.text(.06, .095, '类别2表示预设缺陷刀具下仍正常的切削，不是最严重的异常；若其切削异常，MSM将其归入类别1。', fontsize=10)
    fig.text(.06, .052, '预测单位为已标注切削区间：先对区间内所选窗口的类别分数求均值，再取最大分数类别。', fontsize=10)
    fig.savefig(output, dpi=220, facecolor='white')
    plt.close(fig)


def draw_learning_curves(models: list[dict], output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9))
    fig.subplots_adjust(left=.09, right=.965, top=.80, bottom=.12, hspace=.43, wspace=.25)
    fig.suptitle('训练与验证曲线', x=.06, y=.96, ha='left', fontsize=19)
    fig.text(.06, .90, '仅使用训练集更新参数；以验证集区间级Macro F1选择权重。星号标出各模型选中的epoch。', fontsize=11)
    specifications = [
        ('train_loss', '训练联合损失', 'Loss'),
        ('val_interval_macro_f1', '验证集：区间级Macro F1', 'Macro F1'),
        ('train_ce_loss', '训练分类交叉熵', 'CE loss'),
        ('val_interval_accuracy', '验证集：区间级Accuracy', 'Accuracy'),
    ]
    for ax, (key, title, ylabel) in zip(axes.ravel(), specifications):
        for index, model in enumerate(models):
            observed = [row for row in model['history'] if row.get(key) is not None]
            epochs = [row['epoch'] for row in observed]
            values = [row[key] for row in observed]
            ax.plot(epochs, values, color=COLORS[index], lw=1.6, label=model['name'],
                    marker='o' if key.startswith('val_') else None, markersize=3)
            if key.startswith('val_'):
                best = next(row for row in model['history'] if row['epoch'] == model['report']['best_epoch'])
                ax.scatter([best['epoch']], [best[key]], color=COLORS[index], marker='*', s=135, zorder=4)
        ax.set_title(title, loc='left', fontsize=13)
        ax.set_xlabel('Epoch'); ax.set_ylabel(ylabel)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylim(bottom=0)
        if key.startswith('val_'):
            ax.set_ylim(0, 1.03)
            ax.yaxis.set_major_formatter(PercentFormatter(1))
        style_axes(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(.08, .86), ncol=2, frameon=False)
    fig.text(.06, .047, '仅绘制实际验证点，未评估轮次不补零；周期测试仅观察，验证指标决定选优与early stopping。', fontsize=10)
    fig.savefig(output, dpi=220, facecolor='white')
    plt.close(fig)


def build_html(models, comparison, summary, preprocessing, output):
    result_rows = []
    run_rows = []
    class_rows = []
    window_rows = []
    validation_rows = []
    source_links = []
    baseline_seen = set()
    split_provenance_rows = []
    for model in models:
        name, report = model['name'], model['report']
        metrics = report['test']['interval_metrics']
        support = sum(row['support'] for row in metrics['per_class'])
        result_rows.append([name, pct(metrics['accuracy']), pct(metrics['macro_f1']),
                            pct(metrics['balanced_accuracy']), number(support)])
        baseline = report['majority_baseline_interval_metrics']
        baseline_key = (report['majority_baseline_class'], json.dumps(baseline, sort_keys=True))
        if baseline_key not in baseline_seen:
            baseline_seen.add(baseline_key)
            result_rows.append([f"多数类基线（始终预测类别{report['majority_baseline_class']}）",
                                pct(baseline['accuracy']), pct(baseline['macro_f1']),
                                pct(baseline['balanced_accuracy']), number(sum(row['support'] for row in baseline['per_class']))])
        config = model['config']
        ignored = report.get('ignored_recordings', config.get('ignored_recordings', []))
        split_provenance_rows.append([name,
            report.get('split_csv_source', config.get('split_csv_source', '历史运行未记录CSV来源；以data_split.json为准')),
            ', '.join(ignored) if ignored else '无'])
        run_rows.append([name, report['best_epoch'], report['epochs_run'], number(report['parameter_count']),
                         elapsed(report['training_seconds']), config.get('reconstruction_weight', '未记录'),
                         config.get('beta', '未记录'), report.get('eval_every', config.get('eval_every', 1)),
                         '是' if report.get('test_during_training', config.get('test_during_training', False)) else '否'])
        for row in sorted(metrics['per_class'], key=lambda item: item['class_id']):
            class_rows.append([name, row['class_id'], CLASS_LABELS[row['class_id']], pct(row['precision']),
                               pct(row['recall']), pct(row['f1']), number(row['support'])])
        wm = report['test']['window_metrics']
        window_rows.append([name, pct(wm['accuracy']), pct(wm['macro_f1']),
                            pct(wm['balanced_accuracy']), number(sum(row['support'] for row in wm['per_class']))])
        vm = report['validation']['interval_metrics']
        validation_rows.append([name, pct(vm['accuracy']), pct(vm['macro_f1']),
                                number(sum(row['support'] for row in vm['per_class']))])
        artifacts = [('classification_report.json', '评估JSON'), ('history.json', '训练历史'),
                     ('best_classifier.pt', '分类权重'), ('test_interval_predictions.json', '逐区间测试预测'),
                     ('test_window_predictions.csv', '逐窗口测试预测'),
                     ('dataset_splits_used.csv', '本次划分CSV快照'), ('data_split.json', '本次实际划分'),
                     ('last.pt', '最近完整轮次'), ('final.pt', '最后权重'), ('best.pt', '验证最优检查点')]
        available = [link(model['directory']/filename, label, output) for filename, label in artifacts
                     if (model['directory']/filename).is_file()]
        source_links.append(f'<li><strong>{h(name)}</strong>：' + ' · '.join(available) + '</li>')
    split_rows = []
    first_report = models[0]['report']
    for split, label in [('train', '训练'), ('val', '验证'), ('test', '测试')]:
        counts = models[0]['split_class_counts'][split]
        split_rows.append([label, number(first_report['split_recording_counts'][split]),
                           number(first_report['split_window_counts'][split]),
                           number(counts[0]), number(counts[1]), number(counts[2])])
    comparison_rows = [[row['dimension'], row['at'], row['msm'], row['implication']]
                       for row in comparison['comparison']]
    source_urls = []
    for source in comparison.get('sources', []):
        if source.get('url'):
            source_urls.append(f'<li><a href="{h(source["url"])}">{h(source["id"])}</a></li>')
    fs = preprocessing['target_sample_rate_hz']
    length = preprocessing['sequence_length']
    duration_ms = length/fs*1000
    max_windows = preprocessing['windows_per_interval_limit']
    trim_ms = preprocessing['label_edge_trim_seconds_each_side']*1000
    css = """
    :root{color-scheme:light}body{margin:0;background:#f1f5f8;color:#223548;font:16px/1.75 -apple-system,BlinkMacSystemFont,'PingFang SC','Microsoft YaHei',sans-serif}
    main{max-width:1200px;margin:30px auto;padding:32px 38px;background:#fff;border-radius:14px}h1{font-size:29px;line-height:1.35}h2{margin-top:34px;font-size:22px}a{color:#176789}p{margin:14px 0}
    .lead{font-size:18px}.note{padding:14px 20px;background:#edf5fa;border-left:4px solid #237fa6}.muted{color:#5c6b78;font-size:14px}.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:10px 12px;text-align:left;border-bottom:1px solid #dfe7ed;vertical-align:top}th{background:#edf3f6;white-space:nowrap}img{max-width:100%;height:auto}li{margin-bottom:7px}details{border:1px solid #dbe4eb;border-radius:8px;padding:14px 18px;margin:18px 0}summary{font-weight:600;cursor:pointer}code{background:#f0f4f7;padding:2px 5px}@media(max-width:700px){main{margin:0;padding:20px;border-radius:0}h1{font-size:25px}}
    """
    document = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
    <title>MSM三类状态分类：CNN1D-VAE与LSTM-VAE</title><style>{css}</style></head><body><main>
    <h1>MSM三类状态分类：CNN1D-VAE与LSTM-VAE</h1>
    <p class="lead">两种模型均在已有VAE编码器的潜变量均值μ上增加三类分类头。本页汇总保存的真实训练与测试结果，以完整标注切削区间为主要评估单位。</p>
    <div class="note"><strong>类别定义：</strong>0＝正常切削；1＝需要干预的异常切削；2＝预设缺陷刀具下仍正常的切削。类别2不是最高严重度；缺陷刀具发生异常切削时，MSM标签为1。</div>
    <h2>主要结果：测试切削区间</h2>
    {table(['模型／基线','Accuracy','Macro F1','Balanced accuracy','测试区间数'],result_rows)}
    <p class="muted">同一区间最多{max_windows}个所选窗口的类别分数先求均值，再取最大分数类别。多数类基线只依据训练集窗口类别频数选择固定类别，并在同一测试区间集合上评估。</p>
    <a href="figures/confusion_matrices.png"><img src="figures/confusion_matrices.png" alt="两模型测试区间级混淆矩阵"></a>
    <h2>逐类结果</h2>
    {table(['模型','类别','含义','Precision','Recall','F1','Support：区间数'],class_rows)}
    <p class="muted">Support是测试标注区间数。各区间仍可能来自同一录制，不能视为彼此完全独立的加工试验。模型输出的softmax分数未经概率校准。</p>
    <h2>训练过程与权重选择</h2>
    {table(['模型','最佳epoch','完成epoch编号','参数量','累计训练用时','重构权重','KL权重β','评估间隔epoch','周期观察test'],run_rows)}
    <p>训练目标为分类交叉熵＋重构权重×均方误差＋β×KL。分类始终使用潜变量均值μ；重构分支仅提供辅助训练目标。周期及最后轮评估验证集，最佳权重仅由验证集区间级Macro F1选择。可同时观察测试指标，但它们不参与本程序的权重选择或early stopping；本页最终测试结果来自验证集选出的最佳权重。</p>
    <p class="muted">patience以验证评估次数为单位。未评估轮次的验证值为null，曲线不补零。完整断点恢复保留原history和epoch编号；旧权重续训的历史可从恢复锚点开始，该锚点没有训练损失。</p>
    {table(['模型','最佳权重验证Accuracy','最佳权重验证Macro F1','验证区间数'],validation_rows)}
    <a href="figures/learning_curves.png"><img src="figures/learning_curves.png" alt="训练损失及验证区间级指标曲线"></a>
    <h2>数据划分与预处理</h2>
    <p>预处理缓存包含MSM imi_vm20i的{number(summary['recording_count'])}次录制、{number(summary['labelled_intervals'])}个原始标注切削区间；缓存保留{number(summary['retained_intervals'])}个区间，生成{number(summary['windows'])}个窗口。预处理排除项数为{number(summary['excluded_count'])}，原始Sound0录制总时长为{float(summary['actual_audio_duration_seconds'])/3600:.2f}小时。这些是缓存总量，不是本次CSV选择后的训练样本数。</p>
    <p>本轮实际选择{number(sum(first_report['split_recording_counts'].values()))}次录制、{number(sum(first_report['split_window_counts'].values()))}个窗口。下表以各次运行保存的data_split.json和评估报告为依据，两个模型的具体源录制集合已核对一致。</p>
    {table(['划分','录制数','窗口数','类别0窗口','类别1窗口','类别2窗口'],split_rows)}
    {table(['模型','本次CSV来源','未进入训练／验证／测试的缓存录制'],split_provenance_rows)}
    <ul><li><strong>输入：</strong>Sound0空气声＋Acc0主轴单轴加速度，共同时间网格为{fs/1000:g} kHz；每窗{length}点×2通道，即{duration_ms:.2f}毫秒，float32。</li>
    <li><strong>同步：</strong>Sound0保持48 kHz；Acc0由51.2 kHz按15/16进行抗混叠重采样，再依据时间戳对齐。两路原始单位不同，分别按训练集拟合的通道均值和标准差标准化。</li>
    <li><strong>取窗：</strong>每个标注切削区间两端各留出{trim_ms:g}毫秒，在其内部非重叠完整窗口中均匀选择最多{max_windows}窗，不跨标签边界，不把区间外的背景当作正常切削。</li>
    <li><strong>隔离：</strong>按整次录制划分训练、验证和测试。每次训练读取CSV；exclude及未列入CSV的缓存录制均不参与本次训练、验证或测试。归一化只在本次训练窗口上拟合，划分变更不需要重新采样原始信号。应在训练前确定划分，不根据测试集模型效果反复调整。</li>
    <li><strong>评估：</strong>主要结果采用切削区间级指标；窗口级指标只作为补充。区间级结果仍是稀疏所选窗口的聚合，不等同于连续全时段报警系统的验证。</li></ul>
    <details><summary>补充：窗口级测试指标</summary>
    {table(['模型','窗口Accuracy','窗口Macro F1','窗口Balanced accuracy','窗口数'],window_rows)}
    <p>同一切削区间内的多个窗口相关，窗口数量不能替代独立录制数量；本表不用于选择模型权重。</p></details>
    <h2>MSM与现有AT数据的对照</h2>
    <p class="muted">下表直接读取data_comparison.json中的事实对照，保持原有内容；最终训练输入以本页的实际预处理配置为准。</p>
    {table(['维度','AT现有录制','MSM公开数据','对训练／应用的影响'],comparison_rows)}
    <h2>适用范围</h2>
    <p>本结果属于同一公开机床数据内的录制级隔离实验。同日或相近采集条件的相关性仍可能保留；当前未建立跨日、跨机床或公司DMU50机床的独立泛化结论。类别2只来自少量录制，其支持量和逐类召回应与总体Accuracy一起阅读。</p>
    <p>声音／振动经过重采样和标准化后具有相同的模型输入形状，但不代表两套传感器、加工工况或标签含义等价。公司数据需要先确认实际切削区间及传感器来源，再评估这些分类器是否适用。</p>
    <h2>结果文件与来源</h2><ul>{''.join(source_links)}
    <li>{link(output/'data_comparison.json','数据事实对照JSON',output) if (output/'data_comparison.json').is_file() else '数据对照由命令行指定的JSON提供。'}</li></ul>
    <ul>{''.join(source_urls)}</ul>
    <p class="muted">本页与图件只读取已保存的评估JSON和训练历史；重新生成报告不会训练模型或重新查看测试集以选择参数。</p>
    </main></body></html>"""
    (output/'report.html').write_text(document, encoding='utf-8')


def main():
    here = Path(__file__).resolve().parent
    project = here.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cnn-dir', type=Path, default=project/'CNN1D-VAE/artifacts/msm_classification')
    parser.add_argument('--lstm-dir', type=Path, default=project/'LSTM-VAE/artifacts/msm_classification')
    parser.add_argument('--prepared', type=Path, default=project/'Dataset'/'Data'/'msm_prepared')
    parser.add_argument('--comparison', type=Path, default=here/'data_comparison.json')
    parser.add_argument('--output', type=Path, default=here)
    args = parser.parse_args()
    models = [load_model('CNN1D-VAE', args.cnn_dir), load_model('LSTM-VAE', args.lstm_dir)]
    summary = read_json(args.prepared/'summary.json')
    preprocessing = read_json(args.prepared/'preprocessing.json')
    comparison = read_json(args.comparison)
    if not isinstance(comparison.get('comparison'), list) or not comparison['comparison']:
        raise ValueError('data_comparison.json does not contain comparison rows')
    validate_comparable_models(models)
    args.output.mkdir(parents=True, exist_ok=True)
    figures = args.output/'figures'
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': ['Arial Unicode MS', 'DejaVu Sans'], 'font.size': 11,
                         'axes.unicode_minus': False, 'axes.linewidth': .7,
                         'savefig.facecolor': 'white', 'pdf.fonttype': 42})
    draw_confusion_matrices(models, figures/'confusion_matrices.png')
    draw_learning_curves(models, figures/'learning_curves.png')
    build_html(models, comparison, summary, preprocessing, args.output)
    print(json.dumps({'report': str(args.output/'report.html'),
                      'figures': [str(figures/'confusion_matrices.png'), str(figures/'learning_curves.png')],
                      'models': [model['name'] for model in models]}, ensure_ascii=False))


if __name__ == '__main__':
    main()
