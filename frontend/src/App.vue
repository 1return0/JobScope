<script setup lang="ts">
import { computed } from "vue";
import { RouterLink, RouterView, useRoute } from "vue-router";

const route = useRoute();
const pageTitle = computed(() => String(route.meta.title ?? "JobScope"));
const navigation = [
  { to: "/", label: "运行总览", short: "OV" },
  { to: "/documents", label: "文档工作台", short: "DOC" },
  { to: "/knowledge", label: "检索与问答", short: "RAG" },
  { to: "/agent", label: "Job Agent", short: "AG" },
  { to: "/sources", label: "来源审核", short: "SRC" },
  { to: "/evaluation", label: "质量与评测", short: "EV" },
];
</script>

<template>
  <div class="app-shell">
    <aside class="sidebar">
      <RouterLink class="brand" to="/" aria-label="返回运行总览">
        <span class="brand-mark" aria-hidden="true"><span></span><span></span><span></span></span>
        <span><strong>JobScope</strong><small>TRUSTED JOB INTELLIGENCE</small></span>
      </RouterLink>
      <nav class="primary-nav" aria-label="主导航">
        <p class="nav-label">WORKSPACE</p>
        <RouterLink v-for="item in navigation" :key="item.to" :to="item.to" class="nav-item">
          <span class="nav-icon">{{ item.short }}</span><span>{{ item.label }}</span><span class="nav-arrow">→</span>
        </RouterLink>
      </nav>
      <div class="sidebar-note">
        <span class="signal-dot"></span>
        <div><strong>产品化工作台</strong><p>真实接口、证据边界与实验报告统一呈现。</p></div>
      </div>
    </aside>
    <main class="workspace">
      <header class="topbar">
        <div><p class="topbar-kicker">CANDIDATE-SIDE CAMPUS RECRUITMENT</p><h1>{{ pageTitle }}</h1></div>
        <div class="topbar-actions">
          <a class="text-link" href="http://127.0.0.1:8110/docs" target="_blank">API 文档 →</a>
          <span class="environment-pill">LOCAL · 8110</span>
        </div>
      </header>
      <RouterView />
    </main>
  </div>
</template>
