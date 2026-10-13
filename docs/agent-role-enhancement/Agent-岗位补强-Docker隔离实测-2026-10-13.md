# 固定文档导出 worker 的 Docker 隔离实测

验证日期：2026-10-13。环境：学生虚拟机 `192.168.40.128`。
证据来源：用户在虚拟机手动执行并提供的终端截图；不是助手通过 SSH 执行的日志。

## 实际操作与结果

用户构建 `agent-report-export:local` 镜像，然后执行 `bash verify-isolation.sh`。

| 检查项 | 截图显示结果 |
| --- | --- |
| 固定 worker 生成 DOCX，并解析验证标题 | `export=passed`，36843 字节 |
| 非 root 执行 | `uid=65534` |
| 根文件系统写入被阻止 | `root_write_blocked=true` |
| 网络连接被阻止 | `network_blocked=true` |
| 禁止获取新权限 | `no_new_privileges=true` |
| 有效 capabilities | `effective_capabilities=0` |
| 隔离探针总结果 | `isolation=passed` |

## 可以支撑的结论

固定 JSON-to-DOCX worker 已在真实 Docker 环境生成文件；对应探针验证了非 root、网络限制、根目录写入限制、no-new-privileges 与 capabilities 清除。

运行策略另外配置 CPU、内存、PID 上限和临时目录。这次探针没有进行资源耗尽/超时/逃逸测试，不将「配置了上限」写成「已完成所有资源限额故障测试」。

## 尚不能支撑的结论

- 不能证明一次生产身份 → 真实模型 → PostgreSQL → Docker worker 的应用端到端链路全部通过。
- 截图验证时 Windows 两应用仍使用 `document_export_backend=local`。该手动探针不能证明自动调用；后来独立完成的两项目 `ssh_docker` 实链路见最终验收矩阵，不能反向将后来的结果归到这张截图。
- pgvector 仍为可选后端，原默认 Dense 后端尚未切换；已通过的数据库 Smoke 不代表默认在线问答请求均使用向量库。
- 不证明 Docker 逃逸安全认证、高并发、吞吐、P95/P99 或生产 SLA。
- 这不是任意代码执行功能：worker 不执行模型生成的 Python、Shell 或模板路径。

## 后续验收边界

分项证据已经覆盖真实向量库、文档版本事务、浏览器报告流程与固定 worker 隔离。最终矩阵应分别记录这些结果，以及未验证的完整真实身份/模型/隔离应用链路，不能合并成一次未发生的端到端测试。

保留既有三天补充课程主题与旧游标；本证据记录不新增课程、不修改简历、不 push。
