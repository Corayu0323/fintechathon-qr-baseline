# FinTechathon 赛题五：量价预测研究基础设施

本项目提供本地 D/H/S fixed 研究流程：D 比较候选，H 检验预先冻结的配置，S 生成比赛测试集预测。当前交付不包括 rolling、adaptive、regime 或 aging 模块。

## 本轮冻结方案与结果

- D：2018—2020 拟合、2021—2023 开发比较；按成熟标签时钟修正后，选择 LightGBM + 原值输入，D 综合分 0.117454。
- H：2021—2023 拟合、2024 检验；按冻结方案运行，综合分 0.151572。H 结果不用于事后选型。
- S：2022—2024 拟合、2025—2026 预测；生成 1,599,600 行提交文件，不读取或评分测试标签。

完整 D 候选比较、分项指标和解释见 [QR基础设施交接说明](reports/QR基础设施交接说明.md)。研究假设和时间规则见 `specs/`。

## 数据准备

原始比赛 CSV 不纳入 Git。将 `训练集.csv` 和 `测试集_X.csv` 放入本地 `赛题五数据/`，然后从项目根目录运行：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m scripts.data.audit
.venv/bin/python -m scripts.data.prepare
.venv/bin/python -m scripts.factors.build
.venv/bin/python -m scripts.checks.validate_factors
```

准备过程会在 `data/prepared/` 生成按年份分区的数据，在 `data/features/factor_v0.2/` 生成按版本分区的因子矩阵和 catalog。这些文件已被 `.gitignore` 排除。公式实现与因子元数据分别维护在 `scripts/factors/formulas/` 和 `specs/factor_set_v0.2.json`；同一版本目录不会被重建覆盖，公式变化需提升版本。

## 运行各阶段

从准备好的本地数据运行完整 D 候选比较：

```bash
.venv/bin/python -m scripts.modeling.run_fixed \
  --phase D \
  --models ret1_momentum ret1_reversal ridge lightgbm \
  --transforms train_zscore cs_zscore cs_rank identity
```

之后按固定配置运行 H 和 S，并执行流程与因子验收：

```bash
.venv/bin/python -m scripts.modeling.run_fixed --phase H
.venv/bin/python -m scripts.modeling.run_fixed --phase S
.venv/bin/python -m scripts.checks.validate_pipeline
.venv/bin/python -m scripts.checks.validate_factors
.venv/bin/python -m scripts.checks.validate_model_config
.venv/bin/python -m scripts.factors.diagnostics
```

## 实验记录与看板数据源

每次新运行都会获得独立 run ID 和输出目录，不覆盖同配置的旧模型与预测。参数、D/H 评分、训练规模、运行耗时、模型文件和运行摘要写入本地 Qlib Recorder（MLflow 后端）；预测文件保存在该 run 的本地目录，实验库记录其路径，避免再复制一份大文件。首次接入时，把当前已有 D/H/S 结果导入记录库：

```bash
.venv/bin/python -m scripts.experiments.import_existing_runs
.venv/bin/python -m scripts.reporting.build_dashboard
```

记录库位于 `outputs/experiment_tracking/`，不会进入 Git。可启动 MLflow 页面按阶段、模型、变换和综合分筛选历史运行：

```bash
.venv/bin/mlflow ui \
  --backend-store-uri sqlite:///outputs/experiment_tracking/mlflow.db \
  --host 127.0.0.1 --port 5000
```

之后在浏览器打开 `http://127.0.0.1:5000`。MLflow 是实验记录与筛选页；面向研究汇报的固定数据看板会复用这套记录，不把它做成自由拖拽式 EDA。

研究看板是可离线打开的固定版式，包含 D 综合分候选图、Rank IC/Top 组超额/稳定性分项表、D/H/S 阶段状态及可筛选运行记录。每次新增实验后运行 `.venv/bin/python -m scripts.reporting.build_dashboard` 刷新 `outputs/dashboard/index.html`；也可以直接在浏览器打开该文件。

H/S 的模型和输入变换被代码锁定为 D 选出的 LightGBM + 原值输入。S 产生 `outputs/fixed_baseline/S/submission.csv`，列为 `ts_code,trade_date,pred`。所有预测、模型和重建数据均在 `outputs/` 或 `data/` 的忽略路径下，不应提交到 Git。

## 代码结构

- `scripts/data/`：审计输入数据，并保留原始行、生成标签成熟和行情有效等数据标记。
- `scripts/factors/build.py`：读取版本化因子清单并调度公式函数，按因子集版本原子生成矩阵，不读取标签。
- `scripts/factors/formulas/`：价格形态、收益和成交活跃度公式；每个公式函数独立、无状态。
- `scripts/modeling/preprocessing.py`：独立实现 identity、训练期 z-score、每日截面 z-score 和截面 rank，并保存/复用训练期 scaler 状态。
- `specs/factor_set_v0.2.json`：机器可读的因子清单、输入字段和公式元数据。
- `scripts/checks/validate_factors.py`：核对 key、缺失掩码、数值 parity、未来扰动和分区 warm-up。
- `scripts/factors/diagnostics.py`：在 D 段计算单因子覆盖率、日度 IC/RankIC 汇总和因子间相关性，并记录到 Qlib Recorder；不以单因子指标替代官方模型评分，也不允许把诊断窗口延伸到 H/S。
- `scripts/modeling/qlib_bridge.py`：对齐因子、标签和辅助字段；训练按成熟时点过滤，测试不加载标签。
- `scripts/modeling/run_fixed.py`：D/H/S 阶段配置、拟合、预测、回填和输出。
- `configs/model_recipes.json`：Ridge、LightGBM 和透明信号基线的模型配方；每次运行记录 SHA-256，并把实际配置副本保存到 run 目录与 Recorder。
- `configs/experiment_protocol.json`：D/H/S 日期范围、默认候选和 H/S 冻结方案。
- `scripts/modeling/models/`、`scripts/modeling/config.py`：LightGBM、Qlib Ridge、透明信号预测，以及模型配方和实验方案读取。
- `scripts/checks/`：模型配置、流程、因子和 Qlib 接口的可重复校验。
- `scripts/checks/validate_qlib_bridge.py`：用临时模拟因子核验 Qlib 接口和预测回填，不保留结果。
- `scripts/experiments/`：SQLite 实验库初始化、Qlib Recorder 工具和历史运行导入。
- `scripts/reporting/build_dashboard.py`：从实验记录库生成固定版式、可筛选的本地 HTML 看板。
- `scripts/evaluation/official_score.py`：本地评分适配器。

比赛官方评测中的集合稳定性项是 Jaccard 距离，不是经过成交模拟的实际换手或净收益。D/H 分数均是历史研究结果，S 没有可用测试标签，因此不报告 S 成绩。
