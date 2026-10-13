# 岗位补强最终验收矩阵

更新日期：2026-10-13。本轮冻结开发范围已完成；按下表区分选定实链路与未验证的生产能力。

| 能力 | 实际证据 | 状态与限制 |
| --- | --- | --- |
| JobScope 配置在线 pgvector | 实际默认 Corpus 267 个 Chunk、真实本地 Qwen Embedding、`PgvectorDenseRetriever` 查询返回 3 个 Hit | 本地配置已切换并实测；不是全库检索质量/性能指标 |
| ServiceFlow 配置在线 pgvector | 默认 Corpus 5 个 Chunk、真实模型 Embedding、PostgreSQL 查询返回 3 个 Hit | 本地配置已切换并实测；模型 HTTP 代理改为显式配置，不更改系统代理 |
| 两项目 pgvector 事务、重启、隔离与 stale-reader | 各 4 项真实 PostgreSQL Smoke | 分项通过，固定测试向量 |
| 报告生成/编辑/确认/版本/删除/导出 | 真实 PostgreSQL 与 API 测试、已有浏览器报告 | 通过；部分身份/生成器是 Stub，不是生产身份/模型全链路 |
| Vue 构建与实际操作 | 两项目 build、Edge 生成/预览/确认/刷新/下载/删除 | 通过，页面错误数为 0；Stub 身份/生成器 |
| VM 固定 worker Docker 隔离 | 用户终端截图：export passed、isolation passed、UID 65534 等 | 分项通过；不等于应用自动调用 VM worker |
| Windows API 到 VM Docker | `ssh_docker` 适配器；固定 Docker 命令、正文 stdin、校验 SSH host key | 真实远程导出通过；本地两应用均已切换 ssh_docker |
| JobScope 真实会话/模型到报告/隔离 worker | data/experiments/report-live-20261013.json；2 章节、DOCX 37143 字节 | 选定真实会话、模型、当前 Corpus/pgvector、版本事务与远程 Docker 通过；HTTP 为真实应用的进程内 TestClient |
| ServiceFlow 真实身份/模型到报告/隔离 worker | agent-service/reports/document-real-chain-20261013.json；DOCX 37252 字节，invalid JWT 为 401 | 真实 Java 网络登录/验签、真实模型与 pgvector、报告版本、远程 Docker 通过；当前 Corpus 为演示知识，不能冒充官方校园材料质量验证 |
| 招聘要求对照与三天课程代码位置 | 本目录招聘对照、冻结计划、教学规则及课程游标 | 已整理，开始 Day 1 第 1 课 |

## 当前外部依赖

虚拟机 IP 已从项目配置确认是 `192.168.40.128`，用户名沿用用户截图中的 `root`。SSH 密钥认证此前失败，项目配置未发现 SSH 密码；不尝试拿 PostgreSQL 密码代替。

仓库外配置文件 `tmp/agent-vm.local.env` 的凭据由用户补齐，已验证 SSH；不打印/提交凭据，不读取 MobaXterm 保存的加密凭据，不开放 Docker TCP API，不挂载 Docker socket 给模型工具。

## 最终回归和清理

- JobScope：显式 PG Smoke 配置下全量 Python **455 passed、5 skipped**。
- ServiceFlow：显式 PG Smoke 配置下 Agent 非 MCP 全量 **249 passed、2 skipped**；MCP stdio 单独 **1 passed**。
- 两前端 build 通过。浏览器完整报告操作已有通过记录，但身份/生成器为 Stub；不与进程内真实模型测试合并成一次浏览器真实全链路。
- Java 本轮仅 Maven package（skipTests）与实际登录/验签，不宣称本轮跑了全量 Maven 回归。
- 所有本轮临时 schema、报告、提案、版本、会话/Checkpoint，以及临时 Java QA 账户已清理。启动的 Java 验收进程已停止；VM 既有服务不关闭。
- 仍未验证并发/吞吐/P95、容灾、资源耗尽/逃逸测试、生产 SLA、校园真实材料准确性；已有历史分项证据不伪装成这些结论。

## 学习门槛

本轮教学门槛已经满足，按用户要求采用三天速通集中单元，不覆盖旧周次。2026-10-13 用户明确授权提交并推送 JobScope 与 ServiceFlow；仅发布本轮代码与非敏感证据，不含凭据，不使用 force-push，不修改简历，不恢复 Docling/V2。
