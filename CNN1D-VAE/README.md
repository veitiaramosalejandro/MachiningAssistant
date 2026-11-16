# CNN1D-VAE：双通道信号模型原型

训练新建的5秒WAV四分类数据集，请在项目根目录手动运行：

```bash
python main.py --mode train --task classification --model cnn --csv Dataset/Processed_5s/samples.csv --epochs 200
```

先运行 `Dataset/Preprocessing/prepare_dataset.py`，生成固定目录 `Dataset/Processed_5s/`。每次训练读取指定的单个 `samples.csv`，其中 `split` 列区分训练、验证、测试；每行对应一个完整5秒WAV及其标签。`Dataset/Preprocessing/fourclass_data.py` 负责读取和归一化；保留全部220500帧，末端补44帧以适配卷积，重构损失忽略补齐部分。默认batch size为1，结果保存在 `runs/train/<timestamp>/classification/cnn/`。以下2048点说明属于原有重构任务。

三层一维卷积编码器（通道2→16→32→64，步长均为4），分别预测16维μ和logvar；通过重参数化采样潜变量，三层转置卷积恢复双通道波形。

默认输入是 `(batch, 2048, 2)` float32，采样率44,100 Hz；2048点对应46.44毫秒。两个通道独立保留。模型输出同形状重构波形、潜变量均值和对数方差。训练使用重参数化，评估使用潜变量均值，输出可重复。

本目录位于 `Machining Assistant/CNN1D-VAE`，共享数据位于项目根目录 `Dataset/Data/`，预处理代码和划分表位于 `Dataset/Preprocessing/`。每次调用 `build_dataset` 都重新读取 `Dataset/Preprocessing/dataset_splits.csv`；训练、验证、测试和排除哪些目录由这份CSV决定，代码中不保存固定录制名单。当前CSV已纳入全部31个原始样本目录，分配为25组训练、3组验证、3组测试；新增15段均归训练集，同源片段不跨集合。修改CSV后，下次数据加载即按新划分执行。

CSV列为 `recording_id,source_recording_id,split`。`recording_id` 对应样本目录名；`source_recording_id` 为原始完整录制ID；`split` 填 `train`、`val`、`test` 或 `exclude`。来自同一完整录制的片段必须分到同一个集合。加载器核对 `source_recording_id` 与元数据 `SourceRecording.Folder` 的末级目录名；没有该字段的独立录制使用自身目录名。重复记录、同源跨集合、空集合、活动记录缺文件或来源填写不一致都会报错。`exclude` 和CSV未列出的目录不读取，列入 `ignored_recordings`。

`artifacts/` 保留从原AT项目复制的权重、样例和历史结果；其中旧绝对路径是来源记录。新的训练默认写入项目根目录的 `runs/train/<timestamp>/reconstruction/cnn/`，保存本次实际使用的 `dataset_splits_used.csv`，并在checkpoint信息和测试报告中记录CSV路径、实际划分及忽略目录。修改共享CSV不会改变已有历史结果。

## 已提供的文件

- `model.py`：模型结构和MSE＋beta×KL损失，默认beta=0.001；KL沿潜变量求和后取batch平均。
- `data.py`：轻量导入入口；WAV读取、动态CSV划分、来源核对、窗口选择和归一化统一实现在 `../Dataset/Preprocessing/company_data.py`，两模型共用一份逻辑。
- `runtime.py`：Python推理接口、checkpoint、流式缓冲。
- `run_demo.py`：本模型的训练入口，调用项目根目录共享的 `reconstruction_training.py`。
- `../reconstruction_training.py`：重构训练、周期评估、checkpoint和续训流程。
- `../training_support.py`：时间目录、原子保存及随机状态恢复。
- `predict.py`：WAV输入、JSON输出的命令行入口。
- `tests/`：模型、数据、接口的自动化测试。
- `artifacts/untrained.pt`：未训练的网络权重，附带本次训练窗口拟合的归一化参数。
- `artifacts/smoke_trained.pt`：3轮工程试训权重，状态为 `smoke_trained_not_validated`。
- `artifacts/test_report.json`、`training_history.csv`：实际测试结果和训练记录。
- `artifacts/data_manifest.json`、`window_manifest.json`：数据来源、工况、划分和每个被选窗口的时间。
- `artifacts/window_predictions.csv`：全部256个选定窗口的未训练与试训重构误差。
- `artifacts/example_prediction.json`：包含两窗重构结果的接口样例。
- `artifacts/example_input.wav`：真实数据的连续4096点片段（约93毫秒），便于单独复制本目录后测试；来源记录在 `example_input_source.json`。

## 运行

在本目录执行。已经验证的环境：Python 3.10.19、PyTorch 2.2.2、NumPy 1.26.4；本机解释器是 `/opt/anaconda3/envs/rrg/bin/python`。

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python run_demo.py --data-dir "../Dataset/Data" --split-csv "../Dataset/Preprocessing/dataset_splits.csv" --epochs 3 --windows-per-recording 16 --threads 2
python predict.py --checkpoint artifacts/untrained.pt --wav artifacts/example_input.wav --output runs/prediction.json
```

`predict.py` 默认只处理前32个完整窗口，JSON明确列出已处理数、未处理完整窗数和末尾不足一窗的帧数。`--max-windows 0` 才处理全部完整窗口。`run_demo.py` 可用 `--split-csv` 选择另一份划分表，省略时统一使用项目的 `Dataset/Preprocessing/dataset_splits.csv`，不随 `--data-dir` 改变。`--output-dir` 可以指定另一个新的空结果目录；不指定时使用项目根目录的 `runs/train/<timestamp>/reconstruction/cnn/`。

## 周期评估与断点续训

`run_demo.py` 与根目录 `main.py --mode train --model cnn` 使用同一套共享重构训练流程。从本模型目录开始一次新的200轮训练：

```bash
python run_demo.py --epochs 200 --eval-every 20
```

`--epochs` 表示包含已完成轮数在内的总目标。`--eval-every` 默认20，每20轮同时评估验证集和测试集，最后一轮也会评估。测试结果只用于观察，不参与梯度更新或最佳模型选择；`best.pt` 按实际评估过的验证集重构MSE最小值选择。可加 `--no-test-during-training` 关闭训练期间的周期测试集观察，最终测试评估仍保留。

新结果目录使用 `Europe/Lisbon` 时间，格式为 `YYYY-MM-DD_HH-MM-SS_microseconds`，位于项目的 `runs/train/<timestamp>/reconstruction/cnn/`。显式 `--output-dir` 指定的目录必须不存在或为空；恢复训练也使用新的输出目录，保留来源checkpoint。

每个训练目录保存：

- `last.pt`：每个完整epoch结束后的续训状态。
- `checkpoints/epoch_0020.pt` 等：各评估时点的快照。
- `best.pt`：验证集选出的最佳checkpoint。
- `final.pt`：本次完成时的checkpoint。
- `smoke_trained.pt`：保持现有重构推理接口兼容的最终checkpoint。
- `training_history.csv` 和JSON记录：每轮训练指标及周期验证／测试指标；未评估轮的验证／测试字段为空。

从已完成200轮的 `last.pt` 继续到总计300轮，替换下面的时间目录占位符：

```bash
python run_demo.py --resume "../runs/train/<previous-timestamp>/reconstruction/cnn/last.pt" --epochs 300
```

完整续训恢复模型、优化器、已完成epoch、Python／NumPy／CPU PyTorch及DataLoader generator的随机状态。数据划分、归一化和优化设置需与保存状态一致；可以显式调整总目标轮数、评估间隔和周期测试开关。若在epoch中途停止，该未完成epoch不会保存，续训从最后一个完整epoch之后开始。

旧权重没有完整优化器和随机状态时，仅进行weights-only continuation，并明确记录optimizer reset。保留的 `artifacts/smoke_trained.pt` 是旧3轮工程权重，不应将其视为完整续训checkpoint。这里的断点恢复指训练进度，与流式输入或gRPC连接恢复不同。

项目中已有的旧200轮权重 `../runs/reconstruction/cnn/smoke_trained.pt` 仍在原位置；从该旧文件继续也属于weights-only continuation，不会复现当时的优化器状态。

## Python调用

```python
from data import read_wav
from runtime import VAEEngine, StreamingPredictor

engine = VAEEngine.from_checkpoint("artifacts/untrained.pt")
fs, pcm = read_wav("../Dataset/Data/20261012_124615/audio_000000.wav")
result = engine.predict_window(pcm[:2048], sample_rate=fs)
print(result["reconstruction_mse_standardized"])
print(result["model_status"])

stream = StreamingPredictor(engine)
rows = stream.push(pcm[:1500], sample_rate=fs)  # 不足一个窗口，暂不输出
rows += stream.push(pcm[1500:4096], sample_rate=fs)  # 累计输出两个完整窗口
print(rows[0]["start_s"], rows[0]["end_s"])
```

集成接口接收 **原始PCM16计数**，可以是int16或数值相同的float32；不要先缩放成[-1,1]后再传入。数组形状为单窗 `(2048,2)`、批量 `(B,2048,2)` 或流式块 `(N,2)`。不自动重采样、混合声道或滤除10 kHz成分。采样率不匹配会报错。

`VAEEngine()` 可直接创建不依赖数据文件的未训练实例，默认除以32768。保存的checkpoint使用它自带的通道归一化参数；推理时不重新拟合。重构波形会反变换回PCM计数，保持浮点输出。

流式接口保存不足一窗的尾部，返回每窗的相对起止帧数／秒数。每窗独立建模，LSTM的隐藏状态也按窗重置；跨录制时调用 `stream.reset()`。这是一种分窗流式接口，设备接入、网络传输和公司软件调用方式需在接口约定后接入。

## 输出含义

- `reconstruction_pcm`：重构波形，保留batch维度；默认单窗结果形状为 `(1,2048,2)`。
- `latent_mu`：每窗16维编码。
- `reconstruction_mse_standardized`：标准化空间的重构均方误差，**不是异常概率**。
- `reconstruction_mse_by_channel_standardized`：分别对应L、R的重构误差。
- `model_status`：`untrained` 或 `smoke_trained_not_validated`。
- `validated_for_fault_detection`：当前固定为false；没有经过数据验证的故障类别或报警阈值。

## 保留的历史工程测试

原始16次录制全部参与窗口抽样：10次训练、3次验证、3次测试。每次从不重叠完整窗口网格均匀选择16窗，总计256窗（160/48/48），累计约11.89秒实际信号。它是工程试运行规模，未对全部24分35秒波形进行模型训练。

历史测试按整次录制划分，训练／验证／测试录制ID不交叉。归一化均值和标准差仅用训练窗口拟合，训练、验证和推理复用。当前重新运行时使用CSV中的划分；加入新增数据时，在CSV中为其指定集合和正确原始录制来源，并为新结果选择输出目录。

3轮试训验证损失、梯度和参数更新可以正常运行；同时验证全部选定窗口的未训练推理、权重重载结果、任意长度流式分块与整窗推理一致。报告中还给出始终重构为训练通道均值的简单基线，避免仅凭loss下降判断模型已经有效。

这次历史试训未使用正常／异常标签，原16组来自同日单机且工况混杂。短程试训后的重构改善有限，不能据此判断两种模型的检测性能。右通道约10 kHz固定成分的来源及传感器标定仍需确认。

## 模型接口

```python
from model import ModelConfig, VAE, loss_function

config = ModelConfig(sequence_length=2048, input_channels=2, latent_dim=16)
model = VAE(config)
# x: 已按训练集归一化的float32 Tensor，形状(B,2048,2)
output = model(x)
loss = loss_function(output, x, beta=0.001)
loss["loss"].backward()
```

两个模型目录是独立工程；分别进入对应目录运行上述命令和导入示例。训练权重、归一化参数、窗口长度应作为一个checkpoint共同交付。

## 按实时节奏演示流式输入

`stream_wav.py` 分块读取文件，不等待整段WAV读完。默认每块20 ms（882帧），按块调用现有 `StreamingPredictor`；缓冲满2048帧后输出一个窗口。每个结果立即追加到JSONL文件。

```bash
python stream_wav.py --checkpoint artifacts/smoke_trained.pt --wav "../Dataset/Data/20261012_124615/audio_000000.wav" --chunk-ms 20 --realtime --output runs/stream_predictions.jsonl --report runs/stream_report.json
```

`--realtime` 按信号实际时长回放；去掉它则尽快处理数据块。`--reference-json` 校验是可选项，须先使用当前WAV路径和同一权重通过 `predict.py --max-windows 0` 生成离线结果，再把该结果作为参考。保留的历史JSON记录原项目路径，不直接用于新路径的回放比较。`artifacts/example_input.wav` 可用于不依赖共享数据目录的短演示。

这对应“录制文件离线训练、在线数据分块测试”的模型运行方式。现场数据到达时直接调用 `stream.push(chunk, sample_rate=44100)`，即可替换WAV回放来源。两通道必须已经按时间同步；本模块没有双设备时钟对齐功能，真实振动来源仍需公司确认。

报告中的 `window_end_to_output_*_ms` 从窗口最后一个采样点的理想到达时刻算起，包含20 ms分块带来的等待、调度和Python推理，截止结果就绪；不含JSON写盘及公司软件通信。`emit_push_*_ms` 仅统计产生窗口结果的推理调用耗时。模型预热和加载不在回放计时内。

首个2048点窗口需积累约46.44 ms；在20 ms数据块下通常要等第三块到达，约60 ms后才能输出首个结果。连续采样时余帧会拼入后续窗口，停止回放时不足一窗的部分在报告中记录。

现场C#程序与Python/Linux之间的传输方式、结果字段及加工质量／事件预测目标，应按公司提供的接口要求进一步接入。当前输出仍是未校准的重构误差与编码。

本机单模型实时节奏回放已验证：21.5秒录制以20 ms块输入，共输出462窗，余帧1974；与离线结果在浮点容差内一致。窗口末点到Python结果就绪的中位延迟约14.1 ms，95百分位约23.2 ms。完整记录见 `artifacts/stream_test_report_single_model.json`；先前两模型并行回放记录保留在 `artifacts/stream_test_report.json`。这些是开发机模拟结果，尚未验证公司Linux部署与网络传输。

## MSM三类状态识别

`classifier.py` 在当前VAE的潜变量均值上增加监督分类头，输出正常切削、异常切削、缺陷刀具下的正常切削三类。`train_msm.py` 使用项目根目录 `msm_classification` 中的公共训练流程，`state_runtime.py` 加载专用分类权重。

该流程使用48kHz的Sound0＋Acc0，通道物理单位与原44.1kHz双声道WAV不同，须使用配套预处理。完整数据划分、输入定义及复现命令见 `../msm_classification/README.md`。保留的历史训练结果存于 `artifacts/msm_classification/`，与原重构模型权重分开。

## 保留的历史MSM训练结果

完成28轮训练，依据验证集区间级Macro F1选择第21轮权重。测试集是7次独立于训练/验证的录制，共336个已标注切削区间。

- 区间级Accuracy：75.30%。
- 区间级Macro F1：0.530635。
- 正常／异常／缺陷刀具类召回率：84.34% / 56.00% / 8.33%。
- 多数类基线Accuracy：74.11%。

分类权重：`artifacts/msm_classification/best_classifier.pt`。详细评估：`artifacts/msm_classification/classification_report.json`。已用真实测试窗口验证`StateEngine`加载权重后的输出与评估输出一致，示例在`runtime_example.json`。

区间级成绩是对整段标注切削中所选窗口的分数汇总，不等同每42.67毫秒实时输出的准确率。单窗口成绩另存于测试报告。缺陷刀具测试集只有12个区间、来自1次录制，当前识别仍不足以支持公司现场部署。
