# JobScope项目目录导航

## 1. 分类原则

项目保留Clean Architecture的四个主要层次，并在每层内部按照业务模块继续
分包：

```text
API层 → Application层 → Domain层
              ↑
Infrastructure层实现Application定义的Protocol
```

- `domain`只保存业务对象、业务规则和稳定契约；
- `application`编排用例，并通过Protocol描述需要的外部能力；
- `infrastructure`实现数据库、模型、解析器和报告文件等适配器；
- `api`负责FastAPI请求、响应和生命周期；
- `bootstrap.py`与`runtime_composition.py`负责把实现组装起来。

## 2. app目录

```text
app/
├─ main.py                       FastAPI应用入口
├─ config.py                     环境配置
├─ bootstrap.py                  文档、检索等基础组件工厂
├─ runtime_composition.py        运行期服务组装
├─ api/
│  ├─ documents/                 文档上传与处理接口
│  ├─ retrieval/                 混合检索接口和生命周期
│  └─ answering/                 证据化回答接口
├─ domain/
│  ├─ documents/                 Artifact、Fragment、Chunk、Snapshot、Manifest
│  ├─ sources/                   来源可信度与企业域名
│  └─ jobs/                      证据支持的结构化岗位领域对象
├─ application/
│  ├─ documents/                 解析、切块、Manifest摄取和Corpus审计
│  ├─ retrieval/                 BM25、Dense、RRF、Hybrid和检索评测
│  ├─ answering/                 引用回答、拒答、语义支持与回答评测
│  ├─ ocr/                       OCR回退、组装和准确率评测
│  └─ jobs/                      岗位抽取、校验、规范化和Agent工具
└─ infrastructure/
   ├─ persistence/               SQLAlchemy模型、仓储和查询
   ├─ documents/                 多格式Parser与Manifest文件适配
   ├─ retrieval/                 Embedding模型与评测JSON适配
   ├─ llm/                       OpenAI兼容模型适配器
   ├─ ocr/                       PaddleOCR、PDF渲染和OCR识别适配器
   └─ reports/                   OCR数据集与实验报告序列化
```

## 3. 当前岗位功能定位

| 想找的内容 | 文件 |
| --- | --- |
| 结构化岗位领域对象 | `app/domain/jobs/structured_job_record.py` |
| 岗位抽取工作流 | `app/application/jobs/job_record_extraction_service.py` |
| 截止日期规范化 | `app/application/jobs/job_record_normalization.py` |
| Agent岗位搜索工具 | `app/application/jobs/job_search_tool.py` |
| 当前岗位查询Protocol | `app/application/jobs/current_job_records.py` |
| SQLAlchemy当前岗位查询 | `app/infrastructure/persistence/structured_job_record_query.py` |
| `list_current()` | 上述查询文件中的`SqlAlchemyCurrentJobRecordQuery` |
| 岗位持久化与激活 | `app/infrastructure/persistence/structured_job_record_repository.py` |
| PostgreSQL表模型 | `app/infrastructure/persistence/models.py` |

`SqlAlchemyCurrentJobRecordQuery.list_current()`的完整路径是：

```text
app/infrastructure/persistence/structured_job_record_query.py
```

## 4. tests目录

测试按照被测业务分类，而不是继续把所有测试堆在根目录：

```text
tests/
├─ api/                           FastAPI边界测试
├─ documents/                     文档、Corpus和Parser测试
├─ retrieval/                     检索、融合和性能契约测试
├─ answering/                     回答、引用和拒答测试
├─ ocr/                           OCR解析与评测测试
├─ jobs/                          结构化岗位与Agent工具测试
├─ sources/                       来源可信度和企业域名测试
└─ system/                        应用组装与系统接口测试
```

运行全部测试：

```powershell
py -3.12 -m unittest discover -s tests -v
```

只运行岗位模块：

```powershell
py -3.12 -m unittest discover -s tests/jobs -t . -v
```

## 5. scripts目录

```text
scripts/
├─ corpus/                        Manifest登记、校验、冻结和摄取
├─ sources/                       来源审计与可信域名导入
├─ retrieval/                     检索评测、诊断、搜索和压测
├─ answering/                     回答评测与真实模型Smoke
├─ ocr/                           OCR数据、评测和Smoke
├─ jobs/                          结构化岗位抽取Smoke
└─ reviews/                       每周Word复习生成与批改
```

脚本路径移动后，例如Manifest校验命令为：

```powershell
py -3.12 scripts/corpus/validate_corpus_manifest.py
```

## 6. 新代码放置规则

- 新增纯业务对象：放入对应的`app/domain/<module>/`；
- 新增用例、Protocol或Agent工具：放入`app/application/<module>/`；
- 新增数据库、外部模型或文件实现：放入`app/infrastructure/<module>/`；
- 新增HTTP接口：放入`app/api/<module>/`；
- 测试放入与业务模块同名的`tests/<module>/`；
- 运维和实验命令放入对应的`scripts/<module>/`；
- 不再向`application`、`infrastructure`或`tests`根目录直接堆放业务文件。
