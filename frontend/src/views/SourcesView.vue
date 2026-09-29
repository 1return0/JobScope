<script setup lang="ts">
import { ref } from "vue";
import { sourceApi } from "../api";
import RequestState from "../components/RequestState.vue";
import type { SourceAssessment, SourceCandidate, SourceKind } from "../types";

const candidate = ref<SourceCandidate>({ company: "华为", job_title: "AI Agent 实习生", source_url: "https://career.huawei.com/", source_kind: "official_company" });
const loading = ref(false);
const errorMessage = ref("");
const result = ref<SourceAssessment | null>(null);
const reasonLabels: Record<string, string> = {
  "company-domain-ownership-not-verified": "企业与域名的归属关系尚未被登记表证明",
  "secondary-source-requires-cross-check": "高校来源需要与企业来源交叉复核",
  "aggregated-source-requires-cross-check": "聚合招聘平台需要回到原始来源复核",
  "non-https-url": "传输未使用HTTPS",
  "content-freshness-not-verified": "页面内容的新鲜度尚未验证",
};
const kinds: { value: SourceKind; label: string }[] = [
  { value: "official_company", label: "企业官方" },
  { value: "official_university", label: "高校官方" },
  { value: "job_board", label: "招聘平台" },
];

async function submit() {
  loading.value = true;
  errorMessage.value = "";
  result.value = null;
  try { result.value = await sourceApi.assess(candidate.value); }
  catch (error) { errorMessage.value = error instanceof Error ? error.message : "来源评估失败。"; }
  finally { loading.value = false; }
}
</script>

<template>
  <section class="page-stack">
    <div class="page-intro"><div><span class="eyebrow">SOURCE TRUST</span><h2>区分“格式合法”与“来源可信”</h2><p>HTTPS只证明传输安全；官方域名、内容时效和交叉验证必须分别给出证据。</p></div><span class="boundary-chip">CONSERVATIVE POLICY</span></div>
    <div class="two-column-layout source-layout">
      <form class="panel form-panel" @submit.prevent="submit">
        <div class="field-group"><label for="company">企业名称</label><input id="company" v-model="candidate.company" required maxlength="100" /></div>
        <div class="field-group"><label for="job-title">岗位名称</label><input id="job-title" v-model="candidate.job_title" required maxlength="200" /></div>
        <div class="field-group"><label for="source-url">来源URL</label><input id="source-url" v-model="candidate.source_url" required type="url" maxlength="2048" /></div>
        <div class="field-group"><label>来源类型</label><div class="source-kind-options"><label v-for="kind in kinds" :key="kind.value"><input v-model="candidate.source_kind" type="radio" :value="kind.value" /><span>{{ kind.label }}</span></label></div></div>
        <button class="primary-button" type="submit" :disabled="loading">{{ loading ? "评估中…" : "执行可信度评估" }}</button>
      </form>
      <div class="result-column">
        <RequestState :loading="loading" :error="errorMessage" :empty="!result && !loading && !errorMessage" empty-text="提交来源后会逐项展示域名、传输、时效与人工复核结论。" />
        <template v-if="result">
          <div :class="['trust-verdict', result.trust_verified ? 'success' : 'warning']"><span>{{ result.trust_verified ? "TRUST VERIFIED" : "MANUAL REVIEW" }}</span><strong>{{ result.trust_verified ? "来源已通过当前信任规则" : "不能自动声明该来源可信" }}</strong><p>{{ result.manual_review_required ? "至少一项证据尚未完成，需要人工复核。" : "当前规则要求的证据均已满足。" }}</p></div>
          <div class="trust-grid"><article><span :class="['check-dot', { pass: result.domain_verified }]">{{ result.domain_verified ? "✓" : "×" }}</span><div><strong>企业域名</strong><p>{{ result.matched_official_domain ?? "未匹配登记表" }}</p></div></article><article><span :class="['check-dot', { pass: result.transport_secure }]">{{ result.transport_secure ? "✓" : "×" }}</span><div><strong>传输安全</strong><p>{{ result.transport_secure ? "HTTPS" : "非HTTPS" }}</p></div></article><article><span :class="['check-dot', { pass: result.content_freshness_verified }]">{{ result.content_freshness_verified ? "✓" : "×" }}</span><div><strong>内容时效</strong><p>{{ result.content_freshness_verified ? "已验证" : "尚未验证" }}</p></div></article></div>
          <div class="review-reasons"><span>REVIEW REASONS</span><ul><li v-for="reason in result.review_reasons" :key="reason"><strong>{{ reasonLabels[reason] ?? reason }}</strong><code>{{ reason }}</code></li></ul></div>
        </template>
      </div>
    </div>
  </section>
</template>
