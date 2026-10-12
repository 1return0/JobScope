"""Collect public official recruiting pages into an ignored local snapshot cache."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path

import httpx

from scripts.corpus.normalize_baidu_job_artifact import BaiduJobDomExtractor

ROOT = Path("evaluation/external/recruitment-primary-v1")
EXTRA_JOBS = (
    ("INTERN", "43b42d22-86f9-486b-adf6-7ea633dd4c92"),
    ("INTERN", "f8a467f0-2a4a-4238-a21c-412408f85c2e"),
    ("INTERN", "7e603e30-5367-4474-82fd-eaec9ccc656d"),
    ("INTERN", "0a86720f-d92d-4981-8c8e-a21b24ae13fd"),
    ("INTERN", "a6067048-13d7-4949-ae47-edef93b80b19"),
    ("INTERN", "320200cd-893d-4076-b39d-1f77f7a79948"),
    ("INTERN", "c7c0e663-c7fb-473d-82fa-57a9717d91b2"),
    ("INTERN", "ca7b3bf6-6b8a-4c3f-a8f3-295ab3b9f1f0"),
    ("INTERN", "15f123ac-2755-4324-b5ba-25682d0f9b40"),
    ("GRADUATE", "33f41590-5770-4a91-9395-3c05e7ed1571"),
    ("SOCIAL", "9d3b1a74-c4fb-4ffe-9eff-6124081f70f1"),
    ("SOCIAL", "22f0e1d9-a87d-4c72-868c-17e2176a2b72"),
)
OTHER_SOURCES = {
    "huawei-faq-2026": "https://career.huawei.com/reccampportal/next/mini/faq_h5.html",
    "huawei-campus-2027": "https://career.huawei.com/cn/campus-recruitment",
    "baidu-campus-policy": "https://talent.baidu.com/jobs/campus",
    "baoshan-notice-2026": "https://www.baoshan.gov.cn/info/3281/10350804.htm",
}


def initial_data(html: str) -> dict:
    data = html.split("window.__INITIAL_DATA__ =", 1)[1]
    # Convert only bare JS undefined tokens; quoted text remains untouched.
    data = re.sub(r'"(?:[^"\\]|\\.)*"|\bundefined\b',
                  lambda match: "null" if match.group() == "undefined" else match.group(), data)
    return json.JSONDecoder().raw_decode(data)[0]


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    output = ROOT / "collected-records.json"
    if output.exists():
        raise FileExistsError("source snapshot is already frozen")
    manifest, jobs, candidates = [], [], list(EXTRA_JOBS)
    with httpx.Client(timeout=30, trust_env=False, follow_redirects=True) as client:
        def fetch(source_id, url):
            path = ROOT / (source_id + ".html")
            if not path.exists():
                response = client.get(url)
                response.raise_for_status()
                path.write_bytes(response.content)
            raw = path.read_bytes()
            manifest.append({"source_id": source_id, "url": url,
                             "artifact_sha256": hashlib.sha256(raw).hexdigest(),
                             "byte_size": len(raw)})
            return raw.decode("utf-8")

        for kind in ("INTERN", "GRADUATE", "SOCIAL"):
            html = fetch("baidu-list-" + kind.lower(), "https://talent.baidu.com/jobs/list?recruitType=" + kind)
            listing = initial_data(html)["listData"]
            for row in listing["listDetailData"]:
                # Detail pages carry their own project type; list category is not
                # sufficient evidence for normalizing a record's recruitment type.
                candidates.append((kind, row["postId"]))
        for kind, post_id in dict.fromkeys(candidates):
            url = f"https://talent.baidu.com/jobs/detail/{kind}/{post_id}"
            source_id = f"baidu-{kind.lower()}-{post_id}"
            html = fetch(source_id, url)
            parser = BaiduJobDomExtractor()
            parser.feed(html)
            if not parser.title or len(parser.sections) < 2:
                manifest[-1]["availability"] = "no_public_job_detail"
                print(f"Unavailable job detail: {source_id}", flush=True)
                continue
            detail = initial_data(html)["detailData"]["postInfo"]
            role_id = re.search(r"J\d+", parser.title)
            if role_id is None:
                raise ValueError("missing official role ID")
            jobs.append({"id": role_id.group(), "source_id": source_id,
                         "title": parser.title, "company": "百度", "url": url,
                         "department": detail.get("orgName") or detail.get("bgShortName"),
                         "locations": detail["workPlace"],
                         "recruitment_type_raw": detail["projectType"],
                         "published_at": detail["publishDate"],
                         "sections": dict(parser.sections),
                         "artifact_sha256": manifest[-1]["artifact_sha256"]})
            print(f"Collected {role_id.group()}: {parser.title}", flush=True)
        for source_id, url in OTHER_SOURCES.items():
            fetch(source_id, url)
    payload = {"captured_at": datetime.now(UTC).isoformat(), "jobs": jobs, "sources": manifest}
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Collected {len(jobs)} job pages and {len(manifest)} official snapshots")


if __name__ == "__main__":
    main()
