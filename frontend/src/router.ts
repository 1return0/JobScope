import { createRouter, createWebHistory } from "vue-router";

import AgentView from "./views/AgentView.vue";
import DashboardView from "./views/DashboardView.vue";
import DocumentsView from "./views/DocumentsView.vue";
import EvaluationView from "./views/EvaluationView.vue";
import KnowledgeView from "./views/KnowledgeView.vue";
import SourcesView from "./views/SourcesView.vue";

export const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: "/", name: "dashboard", component: DashboardView, meta: { title: "运行总览" } },
    { path: "/documents", name: "documents", component: DocumentsView, meta: { title: "文档工作台" } },
    { path: "/knowledge", name: "knowledge", component: KnowledgeView, meta: { title: "检索与问答" } },
    { path: "/agent", name: "agent", component: AgentView, meta: { title: "Job Agent" } },
    { path: "/sources", name: "sources", component: SourcesView, meta: { title: "来源审核" } },
    { path: "/evaluation", name: "evaluation", component: EvaluationView, meta: { title: "质量与评测" } },
  ],
});
