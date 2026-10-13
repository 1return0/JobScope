# Agent 岗位补强第二阶段

本页保留第二阶段快照。后续 2026-10-13 已完成真实会话/模型/pgvector/版本事务/SSH Docker 链路，并清理 QA 数据；当前配置已是 pgvector 与 ssh_docker。最新状态以工作区 `docs/Agent-岗位补强-最终验收矩阵.md` 为准，下面旧 local/待验收记录不代表当前状态。

本阶段实现「检索证据 → 文档草稿 → 修改预览 → 确认新版本 → Word 导出」的受控文档办公切片，不是自由执行代码的 Agent。

## 已实现业务

- `POST /v1/job-agent/reports`：由现有可信 Cookie 会话确定 owner，实际调用 Job Agent；只接受注册只读工具的成功结果，不接受浏览器伪造的工具输出或 owner_id。
- 草稿来源：当前岗位结构化事实，以及已有 RAG 的带引用 claims。保留岗位来源范围；无结果/无有效证据不创建空报告。
- 列表、查看、历史版本、删除均在 PostgreSQL 按 application 与 owner 隔离。删除会删除该报告的版本和提案，不涉及 Corpus、Checkpoint 或长期记忆。
- 修改可重命名、选择/重排已有章节、添加用户批注。来源事实和引用不可自由覆盖；批注明确不属于已验证事实。
- 修改提案不改变当前报告；明确 `user_confirmed=true` 才提交新版本。两个旧版本提案竞争只有一个成功，另一个 409；同一提案重复确认返回同一已确认版本。
- 导出指定当前 revision，过期 revision 为 409。固定 worker 接收有大小限制的 JSON，输出 DOCX；不接受模板路径、代码或 Shell。正文、标题、来源和个人批注分区呈现。
- Vue 工作台支持生成、预览、确认、刷新恢复、版本历史、导出和删除。现有匿名会话可复用，不在刷新时强制创建新 owner。

## 核心代码

- `app/application/document_workspace.py`：内容与引用契约、修改规则、招聘结果转报告。
- `app/infrastructure/postgres_report_store.py`：数据库隔离、事务、版本竞争、幂等确认。
- `app/api/jobs/job_report_api.py`：Cookie 主体与现有 Agent 调用入口。
- `app/api/report_workspace_api.py`：文档 CRUD、修改确认与导出 HTTP 链路。
- `app/infrastructure/report_exporter.py`、`report_export_policy.py`、`report_export_worker.py`：固定 worker、输入输出预算、Docker 策略。
- `frontend/src/components/ReportWorkspace.vue`、`views/ReportsView.vue`：前端闭环。

## 验证证据

- 显式连接独立测试库后 Python 全量：**453 passed、5 skipped**。跳过的原有外部 Smoke 不计为成功。
- 文档相关选定单测/真实 PostgreSQL Smoke：**6 passed**。涉及真实 API/数据库/本地子进程导出，但生成器与身份仍有 Stub；不是一次真实登录/模型全链路。
- 前端 `npm run build` 通过。
- 浏览器验收 `scripts/smoke_report_browser.py` 通过：生成、预览、确认 v2、刷新恢复、下载 DOCX、删除；`page_errors=0`。机器报告在 `data/experiments/report-browser-20261012/result.json`。真实 Vue/HTTP/PostgreSQL/导出，身份与生成器为 Stub。
- 真实模型 Planner v3 回归：旧 frozen 契约 **28/32**；Java 实习 title/category 混写的 regression 已修复。余下 4 项参数名契约失败保留原评分，实测岗位集合/引用检查正确。结果在 `evaluation/reports/recruitment-primary-v1-planner-v3-20261012.json`；这仍是开发集回归，不能当独立泛化准确率。

## 运行配置与剩余边界

本地 `.env` 已启用报告存储，使用独立实验 PostgreSQL，不改原招聘库。示例配置默认关闭；启用需设置 `JOBSCOPE_DOCUMENT_REPORTS_ENABLED=true`、`JOBSCOPE_DOCUMENT_REPORTS_DSN`。

`JOBSCOPE_DOCUMENT_EXPORT_BACKEND=local` 是开发模式，只有固定进程与时间/输入预算，**不是 OS 沙箱**。`docker` 后端代码提供无网络、只读根文件系统、非 root、无 capabilities、no-new-privileges、内存/CPU/PID 限制，不挂载主机目录，也不向 worker 传模型/数据库凭据。

2026-10-13 用户在虚拟机执行 `bash verify-isolation.sh` 并提供截图：`export=passed`、36843 字节，`isolation=passed`，UID 65534，根目录写入/网络连接被阻止，no-new-privileges 生效，有效 capabilities 为 0。固定导出 worker 的 Docker 分项验证已通过，不是逃逸安全认证或生产吞吐认证。

本地 API 配置仍为 `local`，没有自动切换到 VM 的 Docker；不把该手动探针描述为真实身份/模型到 Docker 的应用全链路。详细证据位于工作区 `docs/Agent-岗位补强-Docker隔离实测-2026-10-13.md`。

引用完整性不等于模型语义正确性；当前编辑是结构化编辑，不是自由 Word 排版编辑或任意自然语言重写。没有新增训练、自由 Agent 辩论或 Docling/V2。

学习继续按独立三天补充计划；尚未开始正式课。不自动 push 或修改简历。
