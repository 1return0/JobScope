<script setup lang="ts">
import { ref } from "vue";
import { ApiError, retrievalApi } from "../api";
import RequestState from "../components/RequestState.vue";
import type { GroundedAnswerResponse, HybridSearchResponse } from "../types";

const mode = ref<"answer" | "search">("answer");
const query = ref("学校名单里没有我的大学怎么办？");
const topK = ref(5);
const loading = ref(false);
const errorMessage = ref("");
const searchResult = ref<HybridSearchResponse | null>(null);
const answerResult = ref<GroundedAnswerResponse | null>(null);

async function submit() {
  if (!query.value.trim()) {
    errorMessage.value = "请输入招聘相关问题。";
    return;
  }
  loading.value = true;
  errorMessage.value = "";
  searchResult.value = null;
  answerResult.value = null;
  try {
    if (mode.value === "search") searchResult.value = await retrievalApi.search(query.value.trim(), topK.value);
    else answerResult.value = await retrievalApi.answer(query.value.trim(), topK.value);
  } catch (error) {
    errorMessage.value = describeError(error);
  } finally {
    loading.value = false;
  }
}

function describeError(error: unknown) {
  if (error instanceof ApiError && error.status === 503) return "检索或回答服务尚未准备。请启用Hybrid Retrieval，并在回答模式下同时启用Answer Generation。";
  if (error instanceof ApiError && error.status === 429) return "当前并发已满，请稍后重试。";
  if (error instanceof ApiError && error.status === 502) return "模型返回内容未通过结构或引用校验，本次结果已被拒绝。";
  return error instanceof Error ? error.message : "请求失败。";
}

function shortIdentity(value: string) {
  return value.length > 34 ? `${value.slice(0, 31)}…` : value;
}
</script>

<template>
  <section class="page-stack">
    <div class="page-intro"><div><span class="eyebrow">HYBRID RETRIEVAL + GROUNDED RAG</span><h2>检索证据，或生成可验证回答</h2><p>搜索模式展示BM25与Dense经RRF融合后的排名；回答模式只允许引用本次Top-K证据。</p></div><span class="boundary-chip">CURRENT CORPUS ONLY</span></div>
    <form class="query-console" @submit.prevent="submit">
      <div class="segmented mode-tabs"><button type="button" :class="{ active: mode === 'answer' }" @click="mode = 'answer'">证据化回答</button><button type="button" :class="{ active: mode === 'search' }" @click="mode = 'search'">Hybrid检索</button></div>
      <div class="query-row"><textarea v-model="query" maxlength="1000" rows="2" placeholder="输入招聘政策、岗位要求或投递流程问题"></textarea><label>Top-K<input v-model.number="topK" type="number" min="1" max="20" /></label><button class="primary-button" type="submit" :disabled="loading">{{ loading ? "运行中…" : "运行" }}</button></div>
    </form>
    <RequestState :loading="loading" :error="errorMessage" :empty="!searchResult && !answerResult && !loading && !errorMessage" empty-text="提交问题后，这里会显示排名、引用和模型调用边界。" />

    <template v-if="searchResult">
      <div class="trace-strip"><div><span>CORPUS</span><strong>{{ searchResult.corpus_size }}</strong></div><div><span>CANDIDATES</span><strong>{{ searchResult.candidate_k }}</strong></div><div><span>RRF K</span><strong>{{ searchResult.rank_constant }}</strong></div><div><span>EMBEDDING</span><strong :title="searchResult.embedding_identity">{{ shortIdentity(searchResult.embedding_identity) }}</strong></div></div>
      <div v-if="!searchResult.results.length" class="request-state"><span>0</span><strong>未召回证据</strong><p>当前Corpus中没有进入Top-K的结果。</p></div>
      <div class="search-results">
        <article v-for="item in searchResult.results" :key="item.evidence_id" class="ranked-evidence"><div class="rank-badge">{{ item.rank }}</div><div><header><span>{{ item.heading_path.join(" › ") || "未命名章节" }}</span><strong>{{ item.fusion_score.toFixed(6) }}</strong></header><p>{{ item.text }}</p><footer><code>{{ item.evidence_id }}</code><div><span v-for="rank in item.ranks_by_retriever" :key="rank.retriever">{{ rank.retriever }} #{{ rank.rank }}</span></div></footer></div></article>
      </div>
    </template>

    <template v-if="answerResult">
      <div :class="['answer-status', answerResult.status === 'answered' ? 'success' : 'warning']"><div><span>{{ answerResult.status === "answered" ? "GROUNDED ANSWER" : "SAFE REFUSAL" }}</span><strong>{{ answerResult.status === "answered" ? "已生成证据化回答" : "证据不足，系统已拒答" }}</strong></div><div><span>模型调用 {{ answerResult.model_called ? "是" : "否" }}</span><span>引用校验 {{ answerResult.grounding_valid ? "通过" : "未通过" }}</span></div></div>
      <div v-if="answerResult.fallback_message" class="fallback-message">{{ answerResult.fallback_message }}</div>
      <article v-for="(claim, index) in answerResult.claims" :key="index" class="claim-card"><div class="claim-index">{{ String(index + 1).padStart(2, "0") }}</div><div><h3>{{ claim.text }}</h3><div class="citation-list"><details v-for="citation in claim.citations" :key="citation.evidence_id"><summary>{{ citation.heading_path.join(" › ") || citation.source_reference }} · {{ citation.page_number ? `第${citation.page_number}页` : "无页码" }}</summary><blockquote>{{ citation.quoted_text }}</blockquote><code>{{ citation.evidence_id }}</code></details></div></div></article>
      <div class="trace-strip"><div><span>TOP-K</span><strong>{{ answerResult.retrieval.top_k }}</strong></div><div><span>CORPUS</span><strong>{{ answerResult.retrieval.corpus_size }}</strong></div><div><span>FINGERPRINT</span><strong :title="answerResult.retrieval.corpus_fingerprint">{{ shortIdentity(answerResult.retrieval.corpus_fingerprint) }}</strong></div><div><span>GROUNDING</span><strong>{{ answerResult.grounding_valid ? "VALID" : "INVALID" }}</strong></div></div>
    </template>
  </section>
</template>
