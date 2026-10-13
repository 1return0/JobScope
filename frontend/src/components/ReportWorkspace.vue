<script setup lang="ts">
import { onMounted, ref } from "vue";

interface Citation { source_reference: string; evidence_id: string; quoted_text?: string | null }
interface Section { section_id: string; heading: string; body: string; citations: Citation[] }
interface Content { title: string; user_notes: string; sections: Section[] }
interface Report { document_id: string; revision: number; content: Content }
interface Proposal { proposal_id: string; base_revision: number; before: Content; after: Content }
const props = defineProps<{
  basePath: string; baseUrl?: string; tokenProvider?: () => string | null;
  establishSession?: () => Promise<unknown>;
  queryPlaceholder?: string;
}>();
const items = ref<Report[]>([]), current = ref<Report | null>(null);
const history = ref<Report[]>([]), proposal = ref<Proposal | null>(null);
const query = ref(""), title = ref("来源核查报告"), notes = ref("");
const selected = ref<string[]>([]), busy = ref(false), error = ref("");
const exportBackend = ref("disabled");

async function request(path: string, method = "GET", body?: unknown, blob = false) {
  const token = props.tokenProvider?.();
  const response = await fetch(`${props.baseUrl ?? ""}${props.basePath}${path}`, {
    method, credentials: props.tokenProvider ? "omit" : "include",
    headers: { ...(body === undefined ? {} : { "Content-Type": "application/json" }),
      ...(token ? { Authorization: `Bearer ${token}` } : {}) },
    ...(body === undefined ? {} : { body: JSON.stringify(body) }),
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => null);
    const failure = new Error(typeof payload?.detail === "string" ? payload.detail : `HTTP ${response.status}`);
    Object.assign(failure, { status: response.status });
    throw failure;
  }
  return blob ? response.blob() : response.status === 204 ? null : response.json();
}
async function run(operation: () => Promise<void>) {
  busy.value = true; error.value = "";
  try { await operation(); } catch (failure) {
    error.value = failure instanceof Error ? failure.message : "请求失败";
  } finally { busy.value = false; }
}
async function refresh() {
  items.value = (await request("")).items;
  exportBackend.value = (await request("/capabilities")).export_backend;
}
async function open(report: Report) {
  current.value = await request(`/${report.document_id}`);
  title.value = current.value!.content.title; notes.value = current.value!.content.user_notes;
  selected.value = current.value!.content.sections.map(section => section.section_id);
  proposal.value = null;
  history.value = (await request(`/${report.document_id}/revisions`)).items;
}
async function generate() {
  await run(async () => {
    const report = await request("", "POST", { query: query.value, title: title.value });
    await refresh(); await open(report);
  });
}
async function preview() {
  if (!current.value) return;
  await run(async () => {
    proposal.value = await request(`/${current.value!.document_id}/proposals`, "POST", {
      base_revision: current.value!.revision, title: title.value, user_notes: notes.value,
      section_order: selected.value,
    });
  });
}
async function confirm() {
  if (!current.value || !proposal.value) return;
  await run(async () => {
    const report = await request(`/${current.value!.document_id}/proposals/${proposal.value!.proposal_id}/confirm`,
      "POST", { user_confirmed: true });
    await open(report); await refresh();
  });
}
async function download() {
  if (!current.value || !window.confirm("确认导出当前版本？来源引用和用户批注将一并保存。")) return;
  await run(async () => {
    const report = current.value!;
    const data = await request(`/${report.document_id}/export`, "POST", {
      expected_revision: report.revision, user_confirmed: true,
    }, true);
    const url = URL.createObjectURL(data);
    const anchor = document.createElement("a"); anchor.href = url;
    anchor.download = `report-${report.document_id}-v${report.revision}.docx`;
    anchor.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
}
async function remove() {
  if (!current.value || !window.confirm("删除此报告及全部版本和修改提案？此操作不可恢复。")) return;
  await run(async () => {
    await request(`/${current.value!.document_id}`, "DELETE");
    current.value = null; proposal.value = null; history.value = []; await refresh();
  });
}
onMounted(() => run(async () => {
  try { await refresh(); } catch (failure) {
    if ((failure as { status?: number }).status !== 401 || !props.establishSession) throw failure;
    await props.establishSession(); await refresh();
  }
}));
</script>

<template>
  <section class="report-workspace">
    <h2>证据文档工作台</h2>
    <p>从真实查询结果生成报告。可修改标题、选取章节、添加个人批注；原始事实和来源引用不可改写。</p>
    <p>导出模式：{{ exportBackend === 'local' ? '本地固定进程（不是 OS 沙箱）' : ['docker', 'ssh_docker'].includes(exportBackend) ? '配置为 Docker 隔离导出' : '尚未启用' }}</p>
    <p v-if="error" role="alert" class="error">{{ error }}</p>
    <form @submit.prevent="generate">
      <label>报告标题<input v-model="title" maxlength="120" required /></label>
      <label>查询与汇总需求<textarea v-model="query" maxlength="1000" required rows="3" :placeholder="queryPlaceholder ?? '例如：查询指定岗位，再说明其任职要求'" /></label>
      <button :disabled="busy || !query.trim()">生成证据草稿</button>
      <button type="button" :disabled="busy" @click="run(refresh)">刷新列表</button>
    </form>
    <div class="report-columns">
      <aside><h3>我的报告</h3><p v-if="!items.length">暂无报告</p>
        <button v-for="item in items" :key="item.document_id" :disabled="busy" @click="run(() => open(item))">
          {{ item.content.title }} · v{{ item.revision }}
        </button>
      </aside>
      <article v-if="current">
        <h3>{{ current.content.title }} · v{{ current.revision }}</h3>
        <section v-for="section in current.content.sections" :key="section.section_id" class="report-section">
          <label><input v-model="selected" type="checkbox" :value="section.section_id" />保留 {{ section.heading }}</label>
          <p class="body">{{ section.body }}</p>
          <ul><li v-for="citation in section.citations" :key="citation.evidence_id">
            <a v-if="/^https?:\/\//.test(citation.source_reference)" :href="citation.source_reference" target="_blank" rel="noopener noreferrer">查看来源</a>
            <span v-else>项目演示来源（非官方网页）：{{ citation.source_reference }}</span>
            <small>{{ citation.evidence_id }}</small>
            <p v-if="citation.quoted_text">原文：{{ citation.quoted_text }}</p>
          </li></ul>
        </section>
        <label>个人批注（不视为已验证事实）<textarea v-model="notes" maxlength="2000" rows="3" /></label>
        <button :disabled="busy || !selected.length" @click="preview">预览修改</button>
        <button :disabled="busy || exportBackend === 'disabled'" @click="download">确认并导出 Word</button>
        <button :disabled="busy" @click="remove">删除报告</button>
        <section v-if="proposal" class="preview">
          <h4>尚未保存的修改预览 · 基于 v{{ proposal.base_revision }}</h4>
          <p>标题：{{ proposal.before.title }} → {{ proposal.after.title }}</p>
          <p>章节：{{ proposal.before.sections.length }} → {{ proposal.after.sections.length }}</p>
          <p>批注：{{ proposal.after.user_notes || '无' }}</p>
          <button :disabled="busy" @click="confirm">确认保存新版本</button>
          <button :disabled="busy" @click="proposal = null">放弃预览</button>
        </section>
        <h4>版本历史</h4>
        <ul><li v-for="version in history" :key="version.revision">v{{ version.revision }} · {{ version.content.title }}</li></ul>
      </article>
    </div>
  </section>
</template>

<style scoped>
.report-workspace { background: white; padding: 24px; border-radius: 14px; }
form, label { display: grid; gap: 8px; } form { gap: 16px; max-width: 850px; }
input:not([type=checkbox]), textarea { width: 100%; padding: 10px; box-sizing: border-box; border: 1px solid #cad4df; border-radius: 6px; }
button { padding: 9px 14px; margin: 6px 8px 6px 0; cursor: pointer; background: #0b4a78; color: white; border: 1px solid #0b4a78; border-radius: 6px; }
button:disabled { cursor: default; opacity: .55; }
.report-columns { display: grid; grid-template-columns: 250px 1fr; gap: 24px; margin-top: 24px; }
aside button { display: block; width: 100%; text-align: left; }
.report-section, .preview { border-top: 1px solid #dde3eb; padding: 18px 0; }
.report-section label { display: flex; align-items: center; gap: 8px; font-weight: 600; }
input[type=checkbox] { width: 16px; height: 16px; padding: 0; margin: 0; appearance: auto; }
.body { white-space: pre-wrap; } small { display: block; overflow-wrap: anywhere; color: #637387; }
.error { color: #b42318; } @media(max-width: 800px) { .report-columns { grid-template-columns: 1fr; } }
</style>
