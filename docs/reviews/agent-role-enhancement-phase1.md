# Agent 岗位补强第一阶段

日期：2026-10-12。开发尚未全部完成，不启动正式补充课程。

## 本地已实现

- 工具 schema 显式提供 `job_code`，限定 J 加数字；匹配标题中的完整岗位编号，不让 J123 命中 J1234。
- 编号条件与城市/公司等条件共同生效，未知编号返回 `no_matches`；不生成历史记录缺失的岗位编号。
- Planner v2 明确招聘注册/登录/联系 FAQ 是支持场景；不支持的学历硬筛选不能悄悄改成宽查。
- Hybrid 可选 pgvector Dense 工厂；向量复用绑定 corpus 指纹、模型身份、维度与 Chunker 版本。
- 原子替换 PostgreSQL 快照；查询在 PostgreSQL 执行 `<=>` 余弦距离，旧 reader 对已被其他实例替换的快照报不可用。当前是精确搜索，没有 ANN/HNSW 或性能提升宣称。
- 默认仍是原 memory 后端。新增配置 `JOBSCOPE_DENSE_INDEX_BACKEND=pgvector`、`JOBSCOPE_DENSE_POSTGRES_DSN`；缺配置/数据库失败不自动回退。

## 当前验证

- Python 全量：443 passed、8 skipped。新增 3 项真实 pgvector Smoke 因未显式配置 DSN 跳过，不算通过。
- 真实 PostgreSQL 16.14 上原 32 项结构化条件用例：32/32。
- 真实模型仍在原 frozen 32 项任务上运行：严格旧参数契约 27/32，通过率 84.375%。原报告 26/32 不覆盖。
- 独立新报告：`evaluation/reports/recruitment-primary-v1-planner-v2-20261012.json`。没有更改 frozen 输入、答案或检索参数。
- 4 项严格失败实际为旧评测期望 `job_title=Jxxxxx`、新工具使用 `job_code=Jxxxxx`；岗位集合正确，两个混合问答的来源范围与引用检查也正确。保留严格失败，不事后改规则把新数字冒充旧口径。
- 1 项真实 regression：Java 实习查询的模型把 `job_title` 写成「Java实习」，导致无结果。尚待修复及独立保留集验证。
- 本轮问题已用于修复开发，所以该 32 题结果只能作为开发回归，不能声称独立泛化成绩。

## 环境与未完成项

原业务数据库未安装 vector；用户已另行创建独立 `agent-pgvector-lab` 容器，PostgreSQL 16、pgvector 0.8.7，端口 5544。已连接该容器验证，不修改/替换原 PostgreSQL。

真实 pgvector Smoke：`tests/retrieval/test_pgvector_live_smoke.py` **4 passed**，包括数据库余弦排序/索引隔离、失败事务保持旧快照、旧 reader 拒绝替换后的快照、Hybrid 实际 pgvector 适配器重启复用与来源限定。测试使用固定 Stub 向量，不声称真实模型质量或性能提升；临时 schema 已清理。

文档生成/修改/导出闭环、真正的执行隔离、前端联动尚未验收。尚未切换项目默认后端；代码支持不等于本轮所有功能全部完成。

课程计划与教学规则见工作区 `docs/Agent-岗位补强-三天学习计划.md` 和 `docs/Agent-岗位补强-教学规则.md`。尚未 push、尚未修改简历。
