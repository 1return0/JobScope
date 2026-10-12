# BEIR SciFact 外部检索对照

## 为什么新增这组实验

原华为校园招聘 FAQ 集只有 5 个问题、60 个切块，且标注仍为 `provisional`。这些结果继续仅用于诊断，**不得改写为人工已审核，也不得据此宣传招聘场景的检索提升**。

本实验使用 [BEIR 发布的 SciFact test 集](https://github.com/beir-cellar/beir/wiki/Datasets-available)，其查询、语料和相关性标注由数据集提供。它含 5,183 条英文科学文献记录、300 条测试 claim；相关性标注不是本项目生成的。下载包 MD5 为 `5f7d1de60b170fc8027bb7898e2efca1`，与 BEIR 页面一致。

## 复现

1. 从 BEIR 官方页面下载 `scifact.zip`，解压到 `evaluation/external/scifact/`。该目录不入库。
2. 在 JobScope 根目录运行：

   ```powershell
   .venv/Scripts/python.exe -m scripts.retrieval.evaluate_beir_scifact --dataset evaluation/external/scifact --device cuda --batch-size 4 --precision fp16
   ```

3. 查看 `evaluation/reports/beir-scifact-test.json`。脚本校验固定的文档数与测试问题数，并在报告中保留 corpus、queries、qrels 的 SHA-256、Embedding 身份、推理精度、逐题首个相关结果排名。文档与查询向量按块原子写入被忽略的 `evaluation/external/embedding-cache/`，中断后可续跑。

对照只更换检索方法：同一份 title + abstract 语料、同一组 300 条 test claim、同一套官方 qrels、Top-5。BM25 是 JobScope 的 `mixed-cjk-bigram-v1` 实现；Dense 使用项目固定版本的 `Qwen3-Embedding-0.6B`，本地推理采用 FP16 并在报告中注明；Hybrid 用项目的 RRF 实现，两个分支各取 20 个候选、`rank_constant=60`。未用 test 标签调参。FP16 评测不能被叙述为 FP32 线上推理的逐位相同结果。

## 冻结测试集结果

| 方法 | Recall@5 | MRR@5 | nDCG@5 |
| --- | ---: | ---: | ---: |
| JobScope BM25 | 0.694444 | 0.597333 | 0.617099 |
| Dense | **0.790500** | 0.657667 | 0.683504 |
| Hybrid/RRF | 0.773222 | **0.665056** | **0.685927** |

Hybrid/RRF 相比 BM25 分别增加 0.078778、0.067723、0.068828（绝对值）；但 **Recall@5 比 Dense 低 0.017278**，所以不能笼统称为三项指标全面领先。逐题首个相关结果排名对比 Dense：Hybrid 改善 50 题、退化 44 题、持平 206 题。明确退化样例包括 claim ID `1191`、`1196`、`1197`：Dense 在前五名找到标注相关文献，Hybrid 在前五名未找到。逐题首个相关结果排名对比 BM25：改善 62 题、退化 16 题、持平 222 题。完整逐题结果和运行配置见 [机器可读报告](../../evaluation/reports/beir-scifact-test.json)。这些数字只代表这次冻结配置的外部检索实验，不代表上线服务或招聘业务的效果。

按 300 条查询逐题配对、固定随机种子 `20261012` 进行 20,000 次 bootstrap 重采样，Hybrid 相对 BM25 的 MRR@5 差值为 `+0.067722`，95% 区间 `[+0.040721, +0.095222]`。相对 Dense 的差值仅 `+0.007389`，区间 `[-0.025113, +0.039167]`；后者不能作为明确优于 Dense 的证据。该区间衡量此测试集的查询抽样不确定性，不会消除语料领域不匹配等系统偏差。

## 边界

- 这不是 JobScope 招聘语料的检索验收，也不是 Parsing→Chunk→Citation 或 Agent 最终回答评测；BEIR 每条文献在此作为一个检索单位。
- 这是英文科学 claim 检索，Qwen 模型仍沿用 JobScope 的招聘问题 instruction，因而只说明该固定配置在外部数据上的表现。
- 公开 qrels 并非对全部可能相关文献的穷尽标注；MRR、Recall@5 和 nDCG@5 是相对于发布 qrels 计算。
- 原 FAQ 集的 `provisional` 状态不变；若要宣称中文招聘场景的指标，仍需独立人工核验并冻结足量的领域集。
