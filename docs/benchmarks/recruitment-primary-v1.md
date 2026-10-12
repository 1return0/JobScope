# JobScope 招聘领域主评测 v1

## 冻结设计

本评测面向 JobScope 实际业务，替代 SciFact 作为招聘领域的主要质量证据。SciFact 保留为外部英文检索补充实验。

2026-10-12 采集企业官网及政府招聘公告。从可访问的百度公开岗位详情选择40条岗位，涵盖日常实习、校园招聘与社会招聘；另采集华为当前校园招聘页面、百度校园招聘政策页和保山市2026教师招聘公告。只使用实际能读取并核验的43个源页面，原始快照留在忽略入库的本地缓存。未返回职位详情的页面不当作真实岗位，旧版华为FAQ也不混入当前政策资料。

公开数据为40条具体岗位及1条有截止时间的招聘岗位合集公告。由Agent依据原文整理的67个答案/记录单元形成43个标准HTML文档，通过项目原有 HTML Parser 和 StructureAwareChunker 产生337个 Chunk，使用默认 `max_chars=800, overlap_chars=100`。原始网页哈希和派生HTML哈希分别记录，派生摘要不冒充原始全文或逐字原文。问题不进入索引正文。`annotation_status=source_grounded_agent_authored, human_reviewed=false`；来源支持词核验不等于独立人工对全部标注的审核。

在调用评测模型前冻结全部问题、条件与预期结果：

- **134题全局证据检索**：40题岗位职责语义问法、40题指定职位的资格要求、54题投递/考试/入职/毕业范围/截止时间。相关标签指向具体答案段落，不把只命中职位标题或同一网页视为正确。比较实际实现的 BM25、Dense、Hybrid/RRF，统一Top-5。
- **40题指定岗位的SQL范围检索**：使用问题中明示的职位编号经实际 `SearchCurrentJobsTool` 获取来源范围，再用生产 `CurrentCorpusHybridSearchService` 的来源过滤进行检索。与同一40题的无范围Hybrid比较。范围来自用户条件和SQL结果，不来自相关性标签；这组指标单独报告，不能与134题直接混用。
- **32项结构化筛选**：实际SQL和工具执行公司、职位、城市别名、多城市岗位、实习/校招/社招别名、截止日期边界、未注明截止日期及无匹配查询。预期岗位ID集合在跑分前确定。学历/工资/毕业年份不是目前工具的硬筛选字段，另外验证其参数拒绝行为。
- **32项真实Agent任务**：真实Planner、LangGraph、SQL及回答模型执行结构化查询、混合查询、招聘政策问答、无证据与不支持请求。分别记录工具路由、条件保留、返回岗位集合、无证据状态、引用格式与目标证据覆盖。引用检查是结构和事实槽位检查，不冒充独立语义答案正确率。

Dense 使用项目固定版本的 Qwen3-Embedding-0.6B、本地FP32推理，查询指令与线上相同。RRF维持项目当前配置，不用本集标签调参。JobScope当前未实现生产Reranker，本次不另接精排器来制造改善。云端Planner和回答服务使用本地配置的真实模型；响应缓存在与语料、代码及模型配置对应的缓存目录，失败案例仍计入结果。

MRR差值的95%区间按来源页面做配对cluster bootstrap（10,000次、种子20261012），同一岗位或FAQ页面的改写问题一起重采样。日期以固定数据快照为准；毕业范围和发布日期不当作投递截止时间。当前SQL日期字段只支持日粒度，不能证明公告当天15:00之后的小时级截止判定。`current`表示当前材料版本，不自动表示岗位尚可投递。

## 复现

在 JobScope 根目录运行：

```powershell
# 输入校验，不调用模型、不连接数据库。
.venv/Scripts/python.exe -m scripts.jobs.run_recruitment_primary_benchmark --validate-only

# 真实PostgreSQL、本地Embedding与配置的Planner/回答模型。
.venv/Scripts/python.exe -m scripts.jobs.run_recruitment_primary_benchmark
```

PostgreSQL测试在本次随机创建的 `jobscope_benchmark_<uuid>` schema 内进行，完成后只清除此schema。在线Corpus、岗位表、会话Checkpoint和用户偏好数据不被修改。数据库记录由源文核验的固定fixture加载，因此SQL指标不代表模型字段抽取正确率。图评测使用每题新State，不评价多轮记忆或持久化恢复。

源数据重采集及摘要构建分别由 `collect_recruitment_benchmark_sources.py` 和 `build_recruitment_primary_dataset.py` 执行；源页面会变化，完整冻结后的复现使用已提交的 [benchmark.json](../../evaluation/datasets/recruitment-primary-v1/benchmark.json)。完整报告保存为 `evaluation/reports/recruitment-primary-v1-real.json`，已完成报告禁止覆盖。
