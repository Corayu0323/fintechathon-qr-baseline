# QR 基础设施 v1 交接说明

状态：D/H/S fixed 流程已接通；rolling、adaptive、regime 与 aging 研究未纳入 v1。

## 实验记录

反复试验使用 Qlib Recorder + MLflow 的本地 SQLite 后端。每次拟合有独立 run ID；参数、阶段边界、输入变换、评分、训练规模、耗时、模型文件和运行摘要随 run 保存。预测文件保存在唯一的本地 run 目录中，记录库保存其路径以避免复制大型预测文件。实验库与 run 产物位于 `outputs/experiment_tracking/`，不纳入 Git。

接入记录功能前已完成的 D/H/S 结果通过 `scripts/import_existing_runs.py` 导入，标记为 `legacy_import`。该脚本按阶段和旧配置键检查重复，可安全重复运行。之后重跑同一模型/变换会追加新 run；阶段 `summary.json` 作为便于阅读的汇总索引更新，不能代替 Recorder 中不可覆盖的历史记录。

```bash
.venv/bin/python scripts/import_existing_runs.py
.venv/bin/mlflow ui --backend-store-uri sqlite:///outputs/experiment_tracking/mlflow.db --host 127.0.0.1 --port 5000
```

MLflow 页面先承担运行检索和对比；研究型数据看板会从实验记录读取数据，展示预先定义的核心图表与诊断，不采用自由拖拽的 EDA 布局。

固定版式看板由 `scripts/build_dashboard.py` 生成到 `outputs/dashboard/index.html`，不需要额外可视化依赖。它汇总 D 阶段官方综合分及分项对照、H 冻结结果、S 预测覆盖，并支持按阶段、模型、变换筛选最近运行记录及导出 CSV。新实验完成后重新运行构建脚本即可刷新页面；MLflow 页面保留用于查看单个 run 的参数和模型附件。

## 冻结的首轮配置

| 项目 | 配置 |
|---|---|
| 模型 | LightGBM 回归 |
| 输入 | 8 个 `factor_v0.2` 因子，原值输入 |
| 标签 | 原始 `y_ret_1d`，一日收益 |
| 主要选择依据 | 官方综合分：Rank IC 40%、年化 Top 组超额收益 30%、`1 - Jaccard距离` 30% |
| 训练轮数 | 各阶段按预定内部验证流程选择；D/H/S 具体轮数写入该阶段 summary |
| 更新策略 | fixed；预测期内不重训、不替换 |

D 的 8 组配对实验及两个单因子参照信号都已保存在 `outputs/fixed_baseline/summary.json`。按逐次拟合截止修正标签成熟掩码后，D 选择 LightGBM + 原值输入，综合分为 0.117454。该结果仅用于开发选型，不是独立最终成绩。

H 使用 2021—2023 初始化拟合，预测 2024；冻结方案综合分为 0.151572。H 结果不得用于事后改参后再称独立检验。S 使用 2022—2024 拟合，预测 2025—2026；全部 1,599,600 个测试键已输出为 `outputs/fixed_baseline/S/submission.csv`，测试标签在数据接口层不加载、不评分。

D、H、S 是不同训练起点和评估区间，不能直接比较整体分数高低。H 分数高于 D 不足以说明模型泛化“变好”。S 没有本地可见标签，不报告评分。

## D 阶段候选结果

下表用于队友复核 D 内模型与输入变换配对比较。稳定性项按官方公式列为 `1 - mean_turnover`（其中 `mean_turnover` 是每日 Top 10% 集合的 Jaccard 距离）；它不代表实际成交换手。

| 模型/信号 | 输入变换 | Rank IC 均值 | 年化 Top 组超额收益 | 稳定性项 | 综合分 |
|---|---|---:|---:|---:|---:|
| lightgbm | cs_rank | 0.06704 | 0.20860 | 0.08957 | 0.11627 |
| lightgbm | cs_zscore | 0.06220 | 0.15158 | 0.07386 | 0.09251 |
| lightgbm | identity | 0.03872 | 0.20845 | 0.13143 | 0.11745 |
| lightgbm | train_zscore | 0.03958 | 0.20125 | 0.12997 | 0.11520 |
| 1日收益延续参照 | identity | -0.02716 | -0.50191 | 0.08348 | -0.13639 |
| 1日收益反转参照 | identity | 0.03169 | -0.24345 | 0.09884 | -0.03071 |
| ridge | cs_rank | 0.03686 | 0.07314 | 0.08610 | 0.06252 |
| ridge | cs_zscore | 0.04482 | 0.03193 | 0.07268 | 0.04931 |
| ridge | identity | 0.02109 | -0.14878 | 0.06761 | -0.01591 |
| ridge | train_zscore | 0.02110 | -0.14825 | 0.06761 | -0.01575 |

按 D 综合分选择 `lightgbm__identity`，但最优与次优差距较小，应把 D 视为候选筛选而不是最终结论。H/S 已按预先冻结的同一配置运行。

## D 与 H 分数差异诊断

两阶段原始总分不宜直接作优劣比较：D 是固定 2018—2020 模型在 2021—2023 三年上的汇总；H 是新用 2021—2023 拟合的模型在 2024 单年上的评分。D 中训练期 z-score 的 0.11520 与 H 的 0.15157 还使用了不同输入变换。为拆开年份与模型更新的影响，另计算 D 的原值模型在同一 2024 年上的诊断分数；该诊断不改变预先登记的 H 结果。

| 评估段 | 模型拟合历史 | 预测日期 | Rank IC 均值 | 年化 Top 组超额收益 | 稳定性项 | 综合分 |
|---|---|---|---:|---:|---:|---:|
| D 年度诊断 | 2018—2020 | 2021 | 0.04374 | 0.25193 | 0.13762 | 0.13436 |
| D 年度诊断 | 2018—2020 | 2022 | 0.04496 | 0.22842 | 0.13243 | 0.12624 |
| D 年度诊断 | 2018—2020 | 2023 | 0.02744 | 0.14483 | 0.12443 | 0.09175 |
| 同年旧模型诊断 | 2018—2020 | 2024 | 0.05630 | 0.26909 | 0.12249 | 0.13999 |
| H 冻结方案 | 2021—2023 | 2024 | 0.07058 | 0.31192 | 0.09921 | 0.15157 |

2024 年对新旧两版模型都比 2023 年更有利；在同一 2024 年上，H 新模型比 D 旧模型高约 0.01158 分，增量同时来自 Rank IC 和超额收益，Jaccard 稳定性项反而较低。证据支持“年份/市场状态贡献较大，重新拟合可能有额外帮助”的解释，但只有一个 H 年份，无法判断它是可重复规律还是单年现象。季度波动也明显，不能只凭年度总分宣称模型老化被修复或重训必然有效。

## 数据时钟与预处理

- 因子矩阵由 `scripts/build_factors.py` 生成，公式注册在 `scripts/factors/`，定义及版本见 `specs/factor_set_v0.2.json` 与 `data/features/factor_v0.2/catalog.json`；因子生成不读取标签。
- 输入变换已从模型 runner 拆到 `scripts/factor_preprocessing.py`，支持 identity、训练期 z-score、每日截面 z-score、截面 rank；训练期 scaler 可序列化并在验证/预测复用。
- 单因子检查由 `scripts/factor_diagnostics.py` 在 D 段运行，输出覆盖率、日度 IC/RankIC 汇总与因子间日度 Rank 相关，并写入 Qlib Recorder；它只用于因子诊断，模型选型仍按题给综合评分。
- 模型配方移至 `configs/model_recipes.json`，D/H/S 阶段和冻结方案移至 `configs/experiment_protocol.json`；LightGBM、Qlib Ridge 与透明信号基线实现拆在 `scripts/models/`。每次模型运行的配置副本、SHA-256 与模型附件一并写入 Recorder 和本地 run 目录。
- 因子重构验收：新旧 2018—2026 年矩阵列、全量主键和缺失掩码一致，逐列最大绝对差为 0；未来扰动不改变过去因子，按分区加 warm-up 与连续计算一致。旧矩阵原样保存在 `data/features/legacy_unversioned_reference/` 作为本地 parity 参照。
- 每次拟合使用截至该阶段拟合截止日已经成熟的标签。
- 内部验证之前的训练标签，必须在验证开始前最后一个共同交易日收盘时已经成熟。valid 标签只用于选 LightGBM 轮数；最终模型和 scaler 在该阶段全部合格训练样本上重新拟合。
- H/S 锁定模型家族和输入变换。阶段内部仍按同一既定配方从内部 valid 选轮数；具体轮数不人工看 H 评分后调整。
- S 的测试读取不包含 `y_ret_1d` 列；预测保留完整主键，无效输入行用每日低于有效模型预测的保守值填充。

## 复现命令

```bash
.venv/bin/python scripts/build_factors.py
.venv/bin/python scripts/run_fixed_baseline.py --phase D --models ret1_momentum ret1_reversal ridge lightgbm --transforms train_zscore cs_zscore cs_rank identity
.venv/bin/python scripts/run_fixed_baseline.py --phase H
.venv/bin/python scripts/run_fixed_baseline.py --phase S
.venv/bin/python scripts/validate_pipeline.py
```

D 的阶段汇总写入 `outputs/fixed_baseline/summary.json`；H/S 的阶段汇总写入各自子目录。接入 Recorder 后，新运行的预测与模型写到 `outputs/fixed_baseline/<阶段>/runs/<run_uuid>__<配置>/`，同一配置重复运行也会分配新目录；S 的提交文件仍导出到 `outputs/fixed_baseline/S/submission.csv` 作为便捷的最新版本。SQLite 记录库和 MLflow 模型/元数据附件在 `outputs/experiment_tracking/`。这些生成文件均由 `.gitignore` 排除，Git 交接包括代码、spec、因子 catalog、依赖文件与本说明，不包括数据、运行附件或模型。

验收脚本会对照题给 `evaluate.py` 检查评分器、检查 D 候选覆盖、H 冻结边界、S 文件列与全键覆盖，并用一个含伪标签的临时测试分区证明 test 窗口不会加载标签。
