import unittest

from scripts.corpus.normalize_baidu_job_artifact import (
    BaiduJobDomExtractor,
)


class BaiduJobDomExtractorTest(unittest.TestCase):
    def test_extracts_only_targeted_visible_job_blocks(self) -> None:
        extractor = BaiduJobDomExtractor()
        extractor.feed(
            """
            <div class="nav">首页</div>
            <div class="detail-title__abc"><span>大模型实习生</span></div>
            <span class="post-subtitle-item__abc">北京市</span>
            <div class="post-content-title__abc">工作职责：</div>
            <div class="post-content-desc__abc">建设评测流程</div>
            <div class="post-content-title__abc">职责要求：</div>
            <div class="post-content-desc__abc">熟悉Python</div>
            <script>不应被提取</script>
            """
        )
        extractor.close()

        self.assertEqual("大模型实习生", extractor.title)
        self.assertEqual(["北京市"], extractor.metadata)
        self.assertEqual("发布日期", extractor._METADATA_LABELS[-1])
        self.assertEqual(
            [("工作职责", "建设评测流程"), ("职责要求", "熟悉Python")],
            extractor.sections,
        )


if __name__ == "__main__":
    unittest.main()
