# LSTM-VAE：双通道信号模型原型

训练新建的5秒WAV四分类数据集，请在项目根目录手动运行：

```bash
python main.py --mode train --task classification --model lstm --csv Dataset/Processed_5s/samples.csv --lstm-frame-size 441 --epochs 20
```

先运行 `Dataset/Preprocessing/prepare_dataset.py`，生成固定目录 `Dataset/Processed_5s/`。每次训练读取指定的单个 `samples.csv`，其中 `split` 列区分训练、验证、测试；每行对应一个完整5秒WAV及其标签，类别由 `labels.json` 定义。`Dataset/Preprocessing/fourclass_data.py` 负责读取和归一化；`waveform_frames.py` 将连续441个双通道采样点组织成一帧，**保留全部220500个采样点，LSTM递归步数变为500**。每步有882个输入值；解码后仍恢复220500×2的完整波形。训练继续使用CPU，默认batch size为1。

新分类训练默认 `--lstm-frame-size 441`；设置为1可使用旧的逐采样点结构。该值保存在模型配置中，离线和流式推理自动读取。更改它会改变网络层的权重维度，因此新结构需要重新训练；旧checkpoint按保存的设置恢复，缺少该字段时按1处理。结果保存在 `runs/train/<timestamp>/classification/lstm/`。以下2048点说明属于原有重构任务，保留默认frame_size=1。

单层LSTM编码器（hidden=32）从最后状态预测16维μ和logvar；潜变量初始化解码器的隐藏状态和记忆状态，同时作为每步输入，线性层输出双通道波形。解码器不接收目标波形。

默认输入是 `(batch, 2048, 2)` float32，采样率44,100 Hz；2048点对应46.44毫秒。两个通道独立保留。模型输出同形状重构波形、潜变量均值和对数方差。训练使用重参数化，评估使用潜变量均值，输出可重复。

## 共享数据目录与迁移位置

本模型目录位于 `Machining Assistant/LSTM-VAE`，原始录制集中在 `../Dataset/Data`，该目录包含原16组录制和后续新增15组录制。共享预处理代码和划分CSV集中在 `../Dataset/Preprocessing`。本目录的 `data.py` 只转发共享预处理接口，实际读取、取窗、归一化和来源核对逻辑统一保存在 `company_data.py`。默认路径不依赖终端当前目录。

训练、验证、测试集合由共享的 `../Dataset/Preprocessing/dataset_splits.csv` 决定。`build_dataset()` 每次调用都重新读取磁盘上的CSV；`run_demo.py` 每次启动也读取一次，以本轮读取结果贯穿训练和评估，不在代码中固定录制ID。当前CSV使用25组训练、3组验证、3组测试，新增15段均已纳入训练集，同源片段不跨集合；修改CSV后，下次运行即可使用新的划分。

CSV包含 `recording_id,source_recording_id,split` 三列，`split` 只使用 `train`、`val`、`test` 或 `exclude`。`recording_id` 是 `Dataset/Data` 下的样本目录名；`source_recording_id` 是原始录制ID。若元数据有 `SourceRecording.Folder`，必须使用其中原始文件夹的名称；没有该字段的整段录制使用其自身 `recording_id`。同一原始录制拆出的片段必须归入同一个集合，避免来源泄漏。

三个有效集合都必须非空；重复ID、错误来源和已选录制缺失都会报错。`exclude` 和未列入CSV的目录不读取，其名称保存在 `ignored_recordings`。每个窗口和清单条目也记录 `source_recording_id`。更改CSV只影响下一次数据构建，不会改变已保存权重的训练身份。

本目录随迁移保留了既有代码、测试及 `artifacts/` 中的权重和结果，来源为原AT项目的LSTM-VAE。迁移验证只运行测试和推理，不重新训练；已有权重仍保留各自的训练身份和输入约定。新执行 `run_demo.py` 默认把结果写入项目根目录的 `runs/train/<timestamp>/reconstruction/lstm/`。每次调用生成新的时间目录，既有 `artifacts/` 保留原位置。

## 已提供的文件

- `model.py`：模型结构和MSE＋beta×KL损失，默认beta=0.001；KL沿潜变量求和后取batch平均。
- `data.py`：共享预处理的轻量导入入口。
- `../Dataset/Preprocessing/company_data.py`：WAV读取、动态CSV划分、元数据来源核对、窗口选择、训练集通道归一化。
- `../Dataset/Preprocessing/dataset_splits.py`：共享CSV读写和来源泄漏检查。
- `../Dataset/Preprocessing/dataset_splits.csv`：可编辑的训练／验证／测试／排除分配表。
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
- `artifacts/example_input.wav`：真实数据的连续4096点片段（约93毫秒），便于在本项目中进行短时接口测试；来源记录在 `example_input_source.json`。

## 运行

在本目录执行。已经验证的环境：Python 3.10.19、PyTorch 2.2.2、NumPy 1.26.4；本机解释器是 `/opt/anaconda3/envs/rrg/bin/python`。

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python run_demo.py --data-dir "../Dataset/Data" --split-csv "../Dataset/Preprocessing/dataset_splits.csv" --epochs 3 --windows-per-recording 16 --threads 2
python predict.py --checkpoint artifacts/untrained.pt --wav artifacts/example_input.wav --output runs/prediction.json
```

`predict.py` 默认只处理前32个完整窗口，JSON明确列出已处理数、未处理完整窗数和末尾不足一窗的帧数。`--max-windows 0` 才处理全部完整窗口。新训练结果默认保存在项目根目录的 `runs/train/<timestamp>/reconstruction/lstm/`；也可用 `--output-dir` 指定另一个新的空运行目录。省略 `--split-csv` 时统一读取项目中的 `Dataset/Preprocessing/dataset_splits.csv`；即使另传 `--data-dir`，默认CSV位置也不改变，使用其他划分时应显式指定 `--split-csv`。每次运行会保存 `dataset_splits_used.csv` 作为本轮划分快照；checkpoint训练信息和 `test_report.json` 记录CSV绝对路径、实际录制和来源划分、完整CSV行及忽略目录。

## 周期评估与断点续训

`run_demo.py` 与根目录 `main.py --mode train --model lstm` 使用同一套共享重构训练流程。从本模型目录开始一次新的200轮训练：

```bash
python run_demo.py --epochs 200 --eval-every 20
```

`--epochs` 表示包含已完成轮数在内的总目标。`--eval-every` 默认20，每20轮同时评估验证集和测试集，最后一轮也会评估。测试结果只用于观察，不参与梯度更新或最佳模型选择；`best.pt` 按实际评估过的验证集重构MSE最小值选择。可加 `--no-test-during-training` 关闭训练期间的周期测试集观察，最终测试评估仍保留。

新结果目录使用 `Europe/Lisbon` 时间，格式为 `YYYY-MM-DD_HH-MM-SS_microseconds`，位于项目的 `runs/train/<timestamp>/reconstruction/lstm/`。显式 `--output-dir` 指定的目录必须不存在或为空；恢复训练也使用新的输出目录，保留来源checkpoint。

每个训练目录保存：

- `last.pt`：每个完整epoch结束后的续训状态。
- `checkpoints/epoch_0020.pt` 等：各评估时点的快照。
- `best.pt`：验证集选出的最佳checkpoint。
- `final.pt`：本次完成时的checkpoint。
- `smoke_trained.pt`：保持现有重构推理接口兼容的最终checkpoint。
- `training_history.csv` 和JSON记录：每轮训练指标及周期验证／测试指标；未评估轮的验证／测试字段为空。

从已完成200轮的 `last.pt` 继续到总计300轮，替换下面的时间目录占位符：

```bash
python run_demo.py --resume "../runs/train/<previous-timestamp>/reconstruction/lstm/last.pt" --epochs 300
```

完整续训恢复模型、优化器、已完成epoch、Python／NumPy／CPU PyTorch及DataLoader generator的随机状态。数据划分、归一化和优化设置需与保存状态一致；可以显式调整总目标轮数、评估间隔和周期测试开关。若在epoch中途停止，该未完成epoch不会保存，续训从最后一个完整epoch之后开始。

旧权重没有完整优化器和随机状态时，仅进行weights-only continuation，并明确记录optimizer reset。保留的 `artifacts/smoke_trained.pt` 是旧3轮工程权重，不应将其视为完整续训checkpoint。这里的断点恢复指训练进度，与流式输入或gRPC连接恢复不同。

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

## 既有工程试训覆盖的16组数据

`artifacts/` 中现有的历史试训结果使用原始16次录制：10次训练、3次验证、3次测试。每次从不重叠完整窗口网格均匀选择16窗，总计256窗（160/48/48），累计约11.89秒实际信号。它是工程试运行规模，未对全部24分35秒波形进行模型训练。

按整次录制划分，训练／验证／测试录制ID不交叉。归一化均值和标准差仅用训练窗口拟合，训练、验证和推理复用。上述10/3/3属于历史试训记录。当前CSV已更新为25/3/3；从旧断点续训时应通过 `--split-csv` 指定原运行保存的划分快照。后续重新划分时只需编辑CSV；实际集合及窗口数量以当次 `dataset_splits_used.csv` 和报告为准。

3轮试训验证损失、梯度和参数更新可以正常运行；同时验证全部选定窗口的未训练推理、权重重载结果、任意长度流式分块与整窗推理一致。报告中还给出始终重构为训练通道均值的简单基线，避免仅凭loss下降判断模型已经有效。

该历史试训没有使用正常／异常监督标签；所选16组来自同日单机且工况混杂。短程试训后的重构改善有限，不能据此判断两种模型的检测性能。右通道约10 kHz固定成分的来源及传感器标定仍需确认。

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

两个模型分别进入对应目录运行上述命令和导入示例，并共享 `Dataset/Preprocessing` 中的预处理实现和CSV读写模块。交付时保留这一项目结构；训练权重、归一化参数、窗口长度应作为一个checkpoint共同交付。

## 按实时节奏演示流式输入

`stream_wav.py` 分块读取文件，不等待整段WAV读完。默认每块20 ms（882帧），按块调用现有 `StreamingPredictor`；缓冲满2048帧后输出一个窗口。每个结果立即追加到JSONL文件。

```bash
python stream_wav.py --checkpoint artifacts/smoke_trained.pt --wav "../Dataset/Data/20261012_124615/audio_000000.wav" --chunk-ms 20 --realtime --reference-json artifacts/wav_20261012_124615_smoke_trained.json --output runs/stream_predictions.jsonl --report runs/stream_report.json
```

`--realtime` 按信号实际时长回放；去掉它则尽快处理数据块。参考JSON校验是可选项，必须使用同文件、同权重的全窗口离线结果。`example_input.wav` 可用于不依赖原始目录的短演示，此时省略上面的参考JSON参数。

这对应“录制文件离线训练、在线数据分块测试”的模型运行方式。现场数据到达时直接调用 `stream.push(chunk, sample_rate=44100)`，即可替换WAV回放来源。两通道必须已经按时间同步；本模块没有双设备时钟对齐功能，真实振动来源仍需公司确认。

报告中的 `window_end_to_output_*_ms` 从窗口最后一个采样点的理想到达时刻算起，包含20 ms分块带来的等待、调度和Python推理，截止结果就绪；不含JSON写盘及公司软件通信。`emit_push_*_ms` 仅统计产生窗口结果的推理调用耗时。模型预热和加载不在回放计时内。

首个2048点窗口需积累约46.44 ms；在20 ms数据块下通常要等第三块到达，约60 ms后才能输出首个结果。连续采样时余帧会拼入后续窗口，停止回放时不足一窗的部分在报告中记录。

现场C#程序与Python/Linux之间的传输方式、结果字段及加工质量／事件预测目标，应按公司提供的接口要求进一步接入。当前输出仍是未校准的重构误差与编码。

本机单模型实时节奏回放已验证：21.5秒录制以20 ms块输入，共输出462窗，余帧1974；与离线结果在浮点容差内一致。窗口末点到Python结果就绪的中位延迟约19.9 ms，95百分位约29.1 ms。完整记录见 `artifacts/stream_test_report_single_model.json`；先前两模型并行回放记录保留在 `artifacts/stream_test_report.json`。这些是开发机模拟结果，尚未验证公司Linux部署与网络传输。

## MSM三类状态识别

`classifier.py` 在当前VAE的潜变量均值上增加监督分类头，输出正常切削、异常切削、缺陷刀具下的正常切削三类。`train_msm.py` 使用项目根目录 `msm_classification` 中的公共训练流程，`state_runtime.py` 加载专用分类权重。

该流程使用48kHz的Sound0＋Acc0，通道物理单位与原44.1kHz双声道WAV不同，须使用配套预处理。完整数据划分、输入定义及复现命令见 `../msm_classification/README.md`。已完成的历史训练结果保存在 `artifacts/msm_classification/`，与原重构模型权重分开。共享 `Dataset/Data` 的31组录制不等于MSM公开训练集，不会被 `train_msm.py` 自动作为MSM训练输入。

## 保留的既有MSM训练结果

完成17轮训练，依据验证集区间级Macro F1选择第10轮权重。测试集是7次独立于训练/验证的录制，共336个已标注切削区间。

- 区间级Accuracy：81.25%。
- 区间级Macro F1：0.605436。
- 正常／异常／缺陷刀具类召回率：84.34% / 80.00% / 25.00%。
- 多数类基线Accuracy：74.11%。

分类权重：`artifacts/msm_classification/best_classifier.pt`。详细评估：`artifacts/msm_classification/classification_report.json`。已用真实测试窗口验证`StateEngine`加载权重后的输出与评估输出一致，示例在`runtime_example.json`。

区间级成绩是对整段标注切削中所选窗口的分数汇总，不等同每42.67毫秒实时输出的准确率。单窗口成绩另存于测试报告。缺陷刀具测试集只有12个区间、来自1次录制，当前识别仍不足以支持公司现场部署。
