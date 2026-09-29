<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import { evaluationApi } from "../api";
import JsonPanel from "../components/JsonPanel.vue";
import RequestState from "../components/RequestState.vue";
import type { EvaluationArtifactResponse, EvaluationArtifactSummary } from "../types";

const artifacts = ref<EvaluationArtifactSummary[]>([]);
const selected = ref<EvaluationArtifactResponse | null>(null);
const loading = ref(true);
const detailLoading = ref(false);
const errorMessage = ref("");
const typeFilter = ref<"all" | "dataset" | "experiment">("all");
const filtered = computed(() => typeFilter.value === "all" ? artifacts.value : artifacts.value.filter((item) => item.artifact_type === typeFilter.value));
const experimentCount = computed(() => artifacts.value.filter((item) => item.artifact_type === "experiment").length);
const datasetCount = computed(() => artifacts.value.filter((item) => item.artifact_type === "dataset").length);

async function loadCatalog() {
  loading.value = true; errorMessage.value = "";
  try {
    const response = await evaluationApi.catalog();
    artifacts.value = response.artifacts;
    if (artifacts.value.length) await selectArtifact(artifacts.value[0]);
  } catch (error) { errorMessage.value = error instanceof Error ? error.message : "评测目录读取失败。"; }
  finally { loading.value = false; }
}

async function selectArtifact(summary: EvaluationArtifactSummary) {
  detailLoading.value = true;
  try { selected.value = await evaluationApi.artifact(summary.artifact_id); }
  catch (error) { errorMessage.value = error instanceof Error ? error.message : "评测报告读取失败。"; }
  finally { detailLoading.value = false; }
}

function percentage(value: number) { return `${(value * 100).toFixed(value === 1 ? 0 : 1)}%`; }
function displayMetric(name: string) { return name.replaceAll("_", " ").toUpperCase(); }
function formatDate(value: string | null) { return value ? new Date(value).toLocaleString("zh-CN") : "未记录"; }
onMounted(loadCatalog);
</script>

<template>
  <section class="page-stack">
    <div class="page-intro"><div><span class="eyebrow">EVALUATION EVIDENCE</span><h2>让质量结论可追溯，而不是靠一次演示</h2><p>页面只读取冻结黄金集和不可覆盖的实验报告；provisional材料不会被展示成正式结论。</p></div><button class="secondary-button" type="button" @click="loadCatalog">刷新目录</button></div>
    <div class="evaluation-overview"><article><span>DATASETS</span><strong>{{ datasetCount }}</strong><p>OCR、检索、回答、Planner</p></article><article><span>EXPERIMENTS</span><strong>{{ experimentCount }}</strong><p>保留模型、数据哈希与执行时间</p></article><article><span>REVIEW BOUNDARY</span><strong>PROVISIONAL</strong><p>种子数据仍需学生人工复核</p></article></div>
    <RequestState :loading="loading" :error="errorMessage" :empty="!loading && !errorMessage && !artifacts.length" empty-text="data目录中暂时没有评测JSON。" />
    <div v-if="artifacts.length" class="evaluation-layout">
      <aside class="artifact-browser panel"><div class="segmented"><button type="button" :class="{ active: typeFilter === 'all' }" @click="typeFilter = 'all'">全部</button><button type="button" :class="{ active: typeFilter === 'dataset' }" @click="typeFilter = 'dataset'">黄金集</button><button type="button" :class="{ active: typeFilter === 'experiment' }" @click="typeFilter = 'experiment'">实验</button></div><button v-for="artifact in filtered" :key="artifact.artifact_id" type="button" :class="['artifact-item', { active: selected?.summary.artifact_id === artifact.artifact_id }]" @click="selectArtifact(artifact)"><span>{{ artifact.family }} · {{ artifact.artifact_type }}</span><strong>{{ artifact.dataset_id ?? artifact.filename }}</strong><small>{{ artifact.case_count ?? "—" }} cases · {{ artifact.review_status ?? "recorded" }}</small></button></aside>
      <main class="artifact-detail panel">
        <RequestState :loading="detailLoading" :empty="!selected && !detailLoading" />
        <template v-if="selected && !detailLoading"><header><div><span class="eyebrow">{{ selected.summary.artifact_type }}</span><h2>{{ selected.summary.dataset_id ?? selected.summary.filename }}</h2><p>{{ selected.summary.relative_path }}</p></div><span :class="['review-badge', selected.summary.review_status === 'provisional' ? 'warning' : '']">{{ selected.summary.review_status ?? "recorded" }}</span></header><div v-if="Object.keys(selected.summary.metrics).length" class="metric-results"><article v-for="(value, name) in selected.summary.metrics" :key="name"><span>{{ displayMetric(String(name)) }}</span><strong>{{ percentage(value) }}</strong><div><i :style="{ width: `${Math.min(100, Math.max(0, value * 100))}%` }"></i></div></article></div><div class="artifact-metadata"><div><span>CASE COUNT</span><strong>{{ selected.summary.case_count ?? "—" }}</strong></div><div><span>EXECUTED AT</span><strong>{{ formatDate(selected.summary.executed_at) }}</strong></div><div><span>SHA-256</span><code>{{ selected.summary.content_sha256 }}</code></div></div><JsonPanel :value="selected.payload" title="查看完整冻结JSON" /></template>
      </main>
    </div>
  </section>
</template>
