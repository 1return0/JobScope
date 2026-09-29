<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from "vue";
import { RouterLink } from "vue-router";

import { systemApi } from "../api";
import type { HealthResponse, ServiceMetaResponse } from "../types";

const health = ref<HealthResponse | null>(null);
const meta = ref<ServiceMetaResponse | null>(null);
const loading = ref(true);
const errorMessage = ref("");
let controller: AbortController | null = null;

const serviceOnline = computed(() => health.value?.status === "UP");
const stageLabel = computed(() =>
  meta.value?.stage === "productization" ? "产品化组装" : (meta.value?.stage ?? "未知"),
);

const capabilityLabels: Record<string, string> = {
  "scanned-pdf-ocr": "扫描PDF OCR",
  "hybrid-retrieval": "BM25 + Dense + RRF",
  "evidence-grounded-answers": "证据化回答",
  "structured-job-record-extraction": "岗位结构化抽取",
  "job-agent-planning": "Agent Planner",
  "registered-tool-execution": "注册工具执行",
  "planner-evaluation-and-quality-gates": "Planner评测门禁",
  "fastapi-job-agent-runtime-lifecycle": "Runtime生命周期",
  "model-provider-error-translation": "模型异常翻译",
};

const highlightedCapabilities = computed(() => {
  const capabilities = meta.value?.implemented_capabilities ?? [];
  return capabilities
    .filter((item) => capabilityLabels[item])
    .map((item) => capabilityLabels[item]);
});

async function loadOverview() {
  controller?.abort();
  controller = new AbortController();
  loading.value = true;
  errorMessage.value = "";
  try {
    const [healthResponse, metaResponse] = await Promise.all([
      systemApi.health(controller.signal),
      systemApi.meta(controller.signal),
    ]);
    health.value = healthResponse;
    meta.value = metaResponse;
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") return;
    errorMessage.value = error instanceof Error ? error.message : "无法连接JobScope后端";
  } finally {
    loading.value = false;
  }
}

onMounted(loadOverview);
onBeforeUnmount(() => controller?.abort());
</script>

<template>
  <section class="dashboard-page">
    <div v-if="errorMessage" class="connection-banner error-banner">
      <div>
        <strong>后端暂未连接</strong>
        <p>{{ errorMessage }}。请确认FastAPI正在 127.0.0.1:8110 运行。</p>
      </div>
      <button type="button" @click="loadOverview">重新连接</button>
    </div>

    <div v-else class="connection-banner">
      <div class="connection-copy">
        <span :class="['status-light', { online: serviceOnline }]" aria-hidden="true"></span>
        <div>
          <strong>{{ loading ? "正在检查服务" : `${health?.service ?? "JobScope"} 运行正常` }}</strong>
          <p>
            {{ loading ? "读取健康状态与能力契约…" : `${health?.environment} · v${health?.version} · ${stageLabel}` }}
          </p>
        </div>
      </div>
      <span class="live-label">LIVE CONTRACT</span>
    </div>

    <div class="metric-grid" aria-label="系统核心指标">
      <article class="metric-card accent-card">
        <span class="metric-label">SUPPORTED INPUTS</span>
        <strong>{{ meta?.supported_formats.length ?? "—" }}</strong>
        <p>HTML、PDF、Word、Excel、PPT</p>
      </article>
      <article class="metric-card">
        <span class="metric-label">IMPLEMENTED</span>
        <strong>{{ meta?.implemented_capabilities.length ?? "—" }}</strong>
        <p>由后端 /v1/meta 实时提供</p>
      </article>
      <article class="metric-card">
        <span class="metric-label">PLANNED</span>
        <strong>{{ meta?.planned_capabilities.length ?? "—" }}</strong>
        <p>明确标记，避免能力夸大</p>
      </article>
      <article class="metric-card">
        <span class="metric-label">API MODE</span>
        <strong>{{ meta?.read_only ? "R/O" : "R/W" }}</strong>
        <p>当前工作台以预览与查询为主</p>
      </article>
    </div>

    <div class="dashboard-grid">
      <article class="panel architecture-panel">
        <div class="panel-heading">
          <div>
            <span class="eyebrow">SYSTEM FLOW</span>
            <h2>从材料到可信决策</h2>
          </div>
          <span class="panel-index">01</span>
        </div>
        <div class="flow-list">
          <div class="flow-step">
            <span>01</span>
            <div><strong>Document Intelligence</strong><p>多格式解析、OCR、结构感知切块</p></div>
          </div>
          <div class="flow-line"></div>
          <div class="flow-step">
            <span>02</span>
            <div><strong>Current Corpus</strong><p>快照版本、证据边界、当前视图</p></div>
          </div>
          <div class="flow-line"></div>
          <div class="flow-step">
            <span>03</span>
            <div><strong>Hybrid Retrieval</strong><p>BM25与语义检索通过RRF融合</p></div>
          </div>
          <div class="flow-line"></div>
          <div class="flow-step">
            <span>04</span>
            <div><strong>Grounded Agent</strong><p>引用校验、Planner与受控工具执行</p></div>
          </div>
        </div>
      </article>

      <article class="panel capability-panel">
        <div class="panel-heading">
          <div>
            <span class="eyebrow">VERIFIED CAPABILITY</span>
            <h2>当前能力切片</h2>
          </div>
          <span class="panel-index">02</span>
        </div>
        <div class="capability-cloud">
          <span v-for="capability in highlightedCapabilities" :key="capability">
            {{ capability }}
          </span>
          <span v-if="loading" class="skeleton-tag">读取中</span>
        </div>
        <div class="boundary-note">
          <strong>可信边界</strong>
          <p>一次Smoke证明链路可运行；质量结论仍由固定黄金集和多轮实验决定。</p>
        </div>
      </article>
    </div>

    <div class="section-heading">
      <div>
        <span class="eyebrow">WORKBENCH</span>
        <h2>进入业务工作台</h2>
      </div>
      <p>每个页面对应一个已经存在的后端契约。</p>
    </div>

    <div class="workbench-grid">
      <RouterLink to="/documents" class="workbench-card">
        <span class="workbench-code">DOC</span>
        <strong>解析招聘材料</strong>
        <p>上传文件并检查Fragment、Chunk与原始坐标。</p>
        <span class="card-link">打开文档工作台 →</span>
      </RouterLink>
      <RouterLink to="/knowledge" class="workbench-card">
        <span class="workbench-code">RAG</span>
        <strong>检索可信证据</strong>
        <p>查看混合排名、证据原文和带引用回答。</p>
        <span class="card-link">打开检索与问答 →</span>
      </RouterLink>
      <RouterLink to="/agent" class="workbench-card dark-card">
        <span class="workbench-code">AG</span>
        <strong>运行Job Agent</strong>
        <p>观察Planner、工具参数、结果与错误边界。</p>
        <span class="card-link">打开Agent控制台 →</span>
      </RouterLink>
      <RouterLink to="/sources" class="workbench-card">
        <span class="workbench-code">SRC</span>
        <strong>审核招聘来源</strong>
        <p>区分域名归属、HTTPS、时效与人工复核边界。</p>
        <span class="card-link">打开来源审核 →</span>
      </RouterLink>
      <RouterLink to="/evaluation" class="workbench-card">
        <span class="workbench-code">EV</span>
        <strong>查看质量证据</strong>
        <p>读取冻结黄金集、实验指标和完整可追溯JSON。</p>
        <span class="card-link">打开评测中心 →</span>
      </RouterLink>
    </div>
  </section>
</template>
