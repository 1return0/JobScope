# Frozen corpus manifests

运行以下命令后，系统会在本目录生成以Manifest Identity命名的只读语义JSON：

```powershell
py -3.12 scripts\corpus\freeze_corpus_manifest.py
```

冻结文件记录经过验证的清单字段以及实际文件的格式、大小和SHA-256。它不会
复制原始招聘文件；若要长期复现实验，还必须保留`data/corpus-artifacts`中
与哈希对应的原始字节，或将其迁移到不可变对象存储。
