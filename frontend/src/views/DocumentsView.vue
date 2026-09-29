<script setup lang="ts">
import { computed, ref } from "vue";
import { ApiError, documentApi } from "../api";
import RequestState from "../components/RequestState.vue";
import type { DocumentParseResponse, DocumentProcessResponse, EvidenceLocation } from "../types";

const file = ref<File | null>(null);
const sourceReference = ref("local-upload");
const mode = ref<"process" | "parse">("process");
const loading = ref(false);
const errorMessage = ref("");
const result = ref<DocumentParseResponse | DocumentProcessResponse | null>(null);
const dragActive = ref(false);

const accepts = ".html,.htm,.pdf,.docx,.xlsx,.pptx";
const resultItems = computed(() => {
  if (!result.value || result.value.status !== "succeeded") return [];
  return "chunks" in result.value ? result.value.chunks : result.value.fragments;
});

function chooseFile(files: FileList | null) {
  const selected = files?.item(0) ?? null;
  if (selected) {
    file.value = selected;
    result.value = null;
    errorMessage.value = "";
  }
}

function onDrop(event: DragEvent) {
  dragActive.value = false;
  chooseFile(event.dataTransfer?.files ?? null);
}

async function submit() {
  if (!file.value) {
    errorMessage.value = "请先选择一个受支持的文件。";
    return;
  }
  if (!sourceReference.value.trim()) {
    errorMessage.value = "来源标识不能为空。";
    return;
  }
  loading.value = true;
  errorMessage.value = "";
  result.value = null;
  try {
    result.value = mode.value === "process"
      ? await documentApi.process(file.value, sourceReference.value.trim())
      : await documentApi.parse(file.value, sourceReference.value.trim());
  } catch (error) {
    errorMessage.value = describeError(error);
  } finally {
    loading.value = false;
  }
}

function describeError(error: unknown) {
  if (error instanceof ApiError && error.status === 413) return "文件超过后端配置的上传大小限制。";
  if (error instanceof ApiError && error.status === 415) return "文件格式不受支持，或扩展名与真实内容不一致。";
  return error instanceof Error ? error.message : "文档请求失败。";
}

function formatBytes(value: number) {
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
  return `${(value / 1024 / 1024).toFixed(1)} MB`;
}

function locationText(location: EvidenceLocation) {
  const parts = [];
  if (location.heading_path.length) parts.push(location.heading_path.join(" › "));
  if (location.page_number) parts.push(`第 ${location.page_number} 页`);
  if (location.sheet_name) parts.push(`${location.sheet_name}!${location.cell_reference ?? ""}`);
  if (location.slide_number) parts.push(`第 ${location.slide_number} 张幻灯片`);
  if (location.table_number) parts.push(`表格 ${location.table_number}`);
  return parts.join(" · ") || "文档正文";
}
</script>

<template>
  <section class="page-stack">
    <div class="page-intro"><div><span class="eyebrow">MULTI-FORMAT INGESTION</span><h2>把招聘材料转换成可追溯证据</h2><p>支持HTML、PDF、DOCX、XLSX与PPTX；上传内容仅临时解析，本页面不会写入Corpus。</p></div><span class="boundary-chip">TEMPORARY · NO PERSIST</span></div>
    <div class="two-column-layout">
      <form class="panel form-panel" @submit.prevent="submit">
        <div class="field-group"><label>处理模式</label><div class="segmented"><button type="button" :class="{ active: mode === 'process' }" @click="mode = 'process'">解析并切块</button><button type="button" :class="{ active: mode === 'parse' }" @click="mode = 'parse'">仅解析Fragment</button></div></div>
        <label :class="['upload-zone', { active: dragActive }]" @dragover.prevent="dragActive = true" @dragleave.prevent="dragActive = false" @drop.prevent="onDrop">
          <input type="file" :accept="accepts" @change="chooseFile(($event.target as HTMLInputElement).files)" />
          <span class="upload-icon">DOC</span><strong>{{ file?.name ?? "拖入文件或点击选择" }}</strong><p>{{ file ? `${formatBytes(file.size)} · 等待处理` : "HTML / PDF / DOCX / XLSX / PPTX" }}</p>
        </label>
        <div class="field-group"><label for="source-reference">来源标识</label><input id="source-reference" v-model="sourceReference" maxlength="2048" placeholder="例如：https://company.example/careers/123" /><small>用于生成稳定Evidence身份和后续来源追溯。</small></div>
        <button class="primary-button" type="submit" :disabled="loading">{{ loading ? "处理中…" : mode === "process" ? "开始解析与切块" : "开始解析" }}</button>
      </form>

      <div class="result-column">
        <RequestState :loading="loading" :error="errorMessage" :empty="!result && !loading && !errorMessage" empty-text="选择文件后可查看结构化位置、文本片段与稳定Evidence ID。" />
        <template v-if="result">
          <div :class="['result-summary', result.status === 'failed' ? 'danger' : 'success']"><div><span>PROCESS STATUS</span><strong>{{ result.status === "succeeded" ? "处理成功" : "解析失败" }}</strong></div><div class="summary-stats"><span>{{ result.document_format.toUpperCase() }}</span><span>{{ formatBytes(result.byte_size) }}</span><span v-if="'chunk_count' in result">{{ result.chunk_count }} Chunks</span><span v-else>{{ result.fragments.length }} Fragments</span></div></div>
          <div v-if="result.failure" class="failure-card"><strong>{{ result.failure.code }}</strong><p>{{ result.failure.message }}</p><dl><dt>可重试</dt><dd>{{ result.failure.retryable ? "是" : "否" }}</dd><dt>处置建议</dt><dd>{{ result.failure.operator_action }}</dd></dl></div>
          <div v-else class="result-list">
            <article v-for="item in resultItems" :key="'evidence_id' in item ? item.evidence_id : item.ordinal" class="evidence-card">
              <header><span>#{{ 'chunk_ordinal' in item ? item.chunk_ordinal + 1 : item.ordinal + 1 }}</span><strong>{{ locationText(item.location) }}</strong></header>
              <p>{{ item.text }}</p>
              <footer v-if="'evidence_id' in item"><code>{{ item.evidence_id }}</code><span>{{ item.chunker_version }}</span></footer>
            </article>
          </div>
        </template>
      </div>
    </div>
  </section>
</template>
