# 招聘要求对照

依据用户截图可见条款核对，不推测截图外的岗位职责；这是项目证据，不是个人「精通」或录用保证。

| 可见要求 | 项目可展示证据 | 应避免的表述 |
| --- | --- | --- |
| Python/JS、数据库、缓存/MQ、Vue 至少一项 | 两项目 Python/FastAPI 与 Vue/TypeScript；ServiceFlow Java/PostgreSQL/RabbitMQ | 不能因 AI 生成代码就声称个人精通所有技术 |
| Embedding、向量库与 RAG | 默认本地配置 pgvector 实测；BM25/Dense/RRF、ServiceFlow Reranker、来源引用及离线评测 | 不是只把 Checkpoint/Store 叫向量库；不声称 ANN/生产规模性能已验证 |
| Agent 框架、多步规划、记忆 | LangGraph、注册工具、来源限定检索、受控一次重规划、PostgreSQL Checkpoint/长期偏好 | 不包装成任意自治规划、自由 Agent 辩论或多 Worker 生产协调 |
| 全链路、上下文、记忆、沙箱、RAG、MCP | 原 Runtime/context/记忆/Harness；ServiceFlow MCP stdio；固定 Docker worker 与两项目真实 SSH 导出 | MCP 是 ServiceFlow 的能力，不说 JobScope 也已实现；固定 worker 不是任意代码执行平台 |
| 文档办公需求与 Demo、生成/编辑/管理 | 来源证据生成报告、结构化章节编辑/批注、预览确认、版本、刷新恢复、删除与 Word 导出 | 不说完整 Word 排版编辑、任意自然语言重写或成熟办公套件 |
| 全栈 AI 辅助开发加分 | 前后端与测试、版本化交付证据可展示 | 私人 GitHub 仓库不等于外部开源社区贡献；未经许可不代做外部 PR |

当前项目能力已能对应这些核心技术条款，可用于实习岗位演示。个人能力需要能解释真实代码、复现链路与回答取舍；独立开源贡献仍是额外加分项，不通过临时堆功能制造。

验收与限制以 `Agent-岗位补强-最终验收矩阵.md` 为准，特别注意 ServiceFlow 当前在线素材为演示知识、部分浏览器测试使用 Stub、仍无生产 SLA 结论。
