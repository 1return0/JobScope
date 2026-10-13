# 岗位补强课程游标

独立三天模块，不替换原 Week 11/12 或任何 Java 项目周次。

- 当前：速通 Day 1 集中单元「条件/来源限定 + pgvector + 组装边界」。
- 状态：条件与来源部分已讲；不重讲、不要求先答旧三题。继续 pgvector，日末统一三题。
- 下一部分：pgvector 快照与在线检索；Day 1 集中单元三题批改后才推进 Day 2。
- 更新日期：2026-10-13。
- 开始每课前读冻结计划、教学规则、本游标。课程不得重编号、跳课或覆盖旧游标。

## 原九主题代码定位（合并为三天集中单元）

用户已要求速通：原每天三个主题连续讲解，仅每天结束统一三题，不再进行九轮答题。下表保留原主题位置用于找代码，不作为九节独立课程。

| 位置 | 核心代码（app/scripts 等以该项注明的项目或 agent-service 根为准） |
| --- | --- |
| Day 1 第 1 课 | JobScope/app/api/jobs/job_agent_api.py；app/runtime_composition.py；app/application/jobs/job_agent_graph.py、job_agent_planner_prompting.py、job_search_tool.py、job_rag_tools.py；app/infrastructure/persistence/structured_job_record_query.py |
| Day 1 第 2 课 | 两项目的 pgvector_index.py；JobScope/app/infrastructure/retrieval/pgvector_dense_retriever.py；ServiceFlow/agent-service/app/infrastructure/pgvector_support_knowledge_index_store.py |
| Day 1 第 3 课 | JobScope/app/bootstrap.py、config.py；ServiceFlow/agent-service/app/bootstrap.py、config.py、application/support_knowledge_dense_index_runtime.py、mcp/campus_support_knowledge_stdio.py |
| Day 2 第 1 课 | JobScope/app/api/jobs/job_report_api.py、application/document_workspace.py；ServiceFlow/agent-service/app/api/public_knowledge_report_api.py；两项目 app/api/report_workspace_api.py |
| Day 2 第 2 课 | 两项目 app/infrastructure/postgres_report_store.py、application/document_workspace.py |
| Day 2 第 3 课 | 两项目 app/infrastructure/report_exporter.py、ssh_report_exporter.py、report_export_policy.py、report_export_worker.py |
| Day 3 第 1 课 | 原 LangGraph Runtime/context/记忆/Harness/MCP 与新报告生成入口；只讲新增调用点 |
| Day 3 第 2 课 | 两项目 scripts/smoke_configured_report_chain.py、smoke_report_browser.py；pgvector 与 report 的 live_smoke tests；最终验收矩阵 |
| Day 3 第 3 课 | 最终验收矩阵、招聘要求对照；真实演示步骤与面试边界 |

每课讲解必须从真实业务请求出发，展示当前源码和准确可点击链接，解释层次、输入输出、数据字段与分支，最后三题只能考本课已经讲过的内容。
