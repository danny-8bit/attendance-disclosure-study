# 点名配额披露与课堂出勤

在点名课次相同的情况下，公布前后半学期各点几次名，会改变学生的出勤吗？本项目建立一个 **16 节课、恰好点名 10 次**的动态检查博弈，对比只公开全期总次数和同时公开两个阶段配额这两种规则。

实验使用模拟点名表。对 **7,056 张**合法点名表做精确加权，用逆向动态规划求学生的最优响应，并与表格型 Q-learning 的留出测试结果比较。在首次查到缺席只警告的规则下，公开阶段配额对全期出勤率的影响如下。正值表示出勤率提高。

| 上课净成本 | 动态规划 | Q-learning |
| --- | ---: | ---: |
| 0.575 | −3.92 个百分点 | −4.53 个百分点 |
| 0.650 | +7.91 个百分点 | +8.03 个百分点 |

## 复现实验

使用 Python 3.12，在仓库根目录运行以下命令。所有实验均可在 CPU 上运行。

```shell
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python attendance_study.py
python build_report_assets.py
```

快速验证可运行 `python attendance_study.py --quick`，它会执行精确分析和使用两个种子的学习实验，结果写入 `quick_results/`。生成报告图表需要先运行完整实验，将结果写入 `results/`。

## 模型与评估

- 前后半学期各 8 节课。等概率抽取 `4+6`、`5+5`、`6+4` 中的一种配额，再在各阶段内均匀、不放回地抽取点名课次。精确枚举对三组配额赋予相同的总权重，每组内的点名表等权。
- 学生在知道当节是否点名前决定到场或缺席，课后点名结果公开。线性基准对每次查到的缺席扣分。主实验中，第一次被查到只警告，从第二次起每次扣分。
- 学习实验覆盖 7 档上课净成本。每个条件使用 16 个训练种子，每个种子训练 32,768 个模拟学期。两种披露条件共享点名表，冻结后的策略在同一批 4,096 个留出学期上测试。配对置信区间反映不同训练种子之间的差异。

## 文件说明

| 文件 | 用途 |
| --- | --- |
| [`attendance_study.py`](attendance_study.py) | 生成点名表、精确求解、训练 Q-learning 并评估策略 |
| [`build_report_assets.py`](build_report_assets.py) | 从完整实验结果生成矢量图和 LaTeX 表格 |
| [`tests/test_attendance_study.py`](tests/test_attendance_study.py) | 验证点名表权重、贝叶斯更新、信息可见性、扣分和精确基准 |

完整实验会生成以下文件。

- `results/summary.json` 保存理论和学习结果。
- `results/training_curve.csv` 保存各训练检查点的验证指标。
- `results/test_by_seed.csv` 保存各训练种子的留出测试指标。
- `results/raw_results.npz` 保存点名表、Q 表和逐学期测试记录。使用 `numpy.load` 时保留默认的 `allow_pickle=False` 即可读取。
- `figures/` 保存根据完整结果生成的矢量图和表格。

原始数据中的 `state_table` 依次记录课次索引、前半学期已点名次数、后半学期已点名次数、累计被查到缺席次数，以及已公开的前半学期配额。最后一列的 `0` 表示该配额未公开。Q 表中动作 `0` 表示到场，动作 `1` 表示缺席。`test_attendance_mask` 的最低位对应第 1 节课。
