# Qlib 最小接入记录（2026-09-24；2026-09-25 因子工程更新见下）

本轮只完成计划中的环境线与薄数据接口，不生成正式因子，也不以临时烟测数据评价预测能力。

## 接入结构

- 项目 `.venv` 安装 Qlib 0.9.7；Python 3.12.3、NumPy 1.26.4、pandas 2.2.3、PyArrow 23.0.1。直接依赖见 `requirements.txt`，本机完整解析见 `requirements-lock.txt`。未改动全局 Python。
- 已初始化本地 Git 仓库；`.gitignore` 将原始数据、处理后 Parquet、虚拟环境及生成结果排除在跟踪范围之外。当前尚未创建提交或远端。
- `scripts/qlib_bridge.py` 从 `data/prepared/{train,test}_YYYY.parquet` 读取原始标签和辅助标记，从因子集版本目录（例如 `data/features/factor_v0.2/features_YYYY.parquet`）读取 `feature_*`。它拒绝重复键、缺键、多键、跨年因子列不一致和非数值因子。按请求的年份与日期范围读取，不把整套因子先交给 Qlib 再切片。
- `Window.labels` 保留原始 `y_ret_1d`。`label_available` 按标签成熟日计算；`history_ready_L` 按调用方提供的 `min_valid_run` 计算；`model_ready` 同时要求当日行情、历史长度及所有选择的因子有限。训练用的 Qlib label 组才掩去不合格样本，原始标签用于评分。
- `complete_predictions` 将 Qlib 预测按原始键对齐，所有模型不可用行补低于当日有效预测最小值的分数，并检查每条提交键都有有限预测。
- `scripts/official_score.py` 按题给 `evaluate.py` 的三个分项评分，不调用 Qlib 默认交易回测。

## 已完成的真实数据小样本验收

运行 `.venv/bin/python scripts/smoke_qlib.py`：取 2024-12-20 至 2024-12-31 的 37,200 条训练行，以及 2025-01-02 至 2025-01-03 的 9,300 条测试行。临时模拟因子只由股票代码构造，用于检查接口，退出后自动删除。Qlib Ridge 拟合、测试预测与补齐均通过；测试 247 条模型不可用行仍保留键，且低于同日所有正常预测。训练截至 2024-12-31 时，该日 4,525 个原始非空标签没有提前进入训练；将可用时点移至 2025-01-02 后，这些标签才变成可训练。烟测峰值常驻内存约 420 MiB，Python 流程计时约 3.7 秒；这不代表 36 个月训练窗口的内存或耗时。

另外在同一个 37,200 行历史切片上，用固定的非研究性预测列分别运行评分适配器和题给 `evaluate.py`，返回的 8 个指标均在 `1e-12` 容差内一致。这个检查只验证评分实现，不代表样本外表现。

## 2026-09-25 因子工程更新

首轮因子现已由版本清单 `specs/factor_set_v0.2.json` 驱动，纯公式按收益、价格形态、成交活跃度拆在 `scripts/factors/`；结果保存在 `data/features/factor_v0.2/`，构建采用新版本目录，不覆盖同版本已有结果。旧的未分版本矩阵保留在 `data/features/legacy_unversioned_reference/`。

`.venv/bin/python scripts/validate_factors.py` 已核对 2018—2026 年全部九个分区：新旧 key、列次序、缺失位置完全一致，8 列最大数值差均为 0；合成数据的未来扰动和分区 warm-up 检查通过。旧段落中的“正式因子工程需输出”和“实验记录尚未启动”描述是 2026-09-24 当时状态，已被本次更新替代。

## 2026-09-24 下一道门槛（历史状态）

正式因子工程需输出年度 `features_YYYY.parquet` 和 `feature_catalog`，并明确最长连续有效日线需求 `min_valid_run`、列顺序、版本与缺失规则。随后才做真实因子的一次 36 个月窗口加载和峰值内存测试，再进入滚动训练。Qlib 的实验记录和正式模型比较尚未启动；不应把上述模拟因子的拟合结果纳入研究报告。
