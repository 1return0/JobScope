# 发布前检查

日期：2026-10-13。用户明确授权将本轮新增代码、非敏感验收记录及三天速通课程文档提交并推送。

## 本次重新验证

- JobScope：普通 Python 回归 447 passed、13 skipped。
- ServiceFlow：Agent 非 MCP 回归 241 passed、10 skipped；MCP stdio 单独 1 passed。
- 两个 Vue 前端 `npm run build` 通过。
- Git staged diff 格式检查通过；凭据扫描没有发现实际 API Key、SSH 密码/私钥、真实带密码 DSN 或私有 .env 文件。泛用示例、测试假凭据不作为真实凭据。

当前独立 VM 的 pgvector 端口 5544 不可达。本次显式真实数据库重跑出现连接错误并停止等待，随后普通回归将外部 Smoke 跳过；**不把本次外部 Smoke 记为 passed**。

此前选定真实 PG/模型/SSH-Docker 链路的报告仍保留，是历史实测证据，不改写成今天这次连接失败后的新测试结果。总体能力和未验证边界见最终验收矩阵。

## 提交边界

只发布本轮实现、必要测试、Docker worker、前端、README、非敏感报告/截图和课程文档。不包括本地 .env、仓库外 SSH 配置、venv、模型、缓存、临时压缩包。

ServiceFlow 原有 `docs/COURSE_CURSOR_DELIVERY_ENGINEERING.md` 删除改动属于用户，本次不纳入、不恢复。CampusReserve 和简历不修改。不使用 force-push，不改写远端历史。
