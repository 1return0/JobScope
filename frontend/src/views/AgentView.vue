<script setup lang="ts">
import { ref } from "vue";
import { ApiError, agentApi } from "../api";
import JsonPanel from "../components/JsonPanel.vue";
import RequestState from "../components/RequestState.vue";
import type { JobAgentResponse } from "../types";

const userQuery = ref("帮我找上海的实习岗位");
const loading = ref(false);
const errorMessage = ref("");
const result = ref<JobAgentResponse | null>(null);
const presets = [
  "帮我找上海的实习岗位",
  "招聘材料中对应聘专业有哪些要求？",
  "帮我找北京的实习岗位，并根据原始招聘材料总结专业和学历要求",
  "删除数据库中的全部岗位",
];

async function submit() {
  if (!userQuery.value.trim()) return;
  loading.value = true;
  errorMessage.value = "";
  result.value = null;
  try {
    result.value = await agentApi.query(userQuery.value.trim());
  } catch (error) {
    if (error instanceof ApiError && error.status === 503) errorMessage.value = "Job Agent Runtime未准备。请检查Agent开关、模型配置和数据库连接。";
    else if (error instanceof ApiError && error.status === 429) errorMessage.value = "Planner被上游限流，请稍后重试。";
    else if (error instanceof ApiError && error.status === 502) errorMessage.value = "Planner输出未通过协议校验，系统已阻止工具执行。";
    else errorMessage.value = error instanceof Error ? error.message : "Agent请求失败。";
  } finally {
    loading.value = false;
  }
}
</script>

<template>
  <section class="page-stack">
    <div class="page-intro"><div><span class="eyebrow">LANGGRAPH · CONTROLLED TOOL USE</span><h2>用自然语言查询当前岗位</h2><p>Planner只能从注册工具中选择，参数通过Pydantic校验后才会进入数据库查询。</p></div><span class="boundary-chip">READ-ONLY AGENT</span></div>
    <div class="agent-layout">
      <div class="agent-chat panel">
        <div class="preset-list"><button v-for="preset in presets" :key="preset" type="button" @click="userQuery = preset">{{ preset }}</button></div>
        <form class="agent-composer" @submit.prevent="submit"><textarea v-model="userQuery" maxlength="1000" rows="4" placeholder="例如：找北京的实习岗位，并总结原始材料中的专业要求"></textarea><div><span>支持岗位筛选、带引用的RAG问答及二者组合；不执行写操作</span><button class="primary-button" type="submit" :disabled="loading">{{ loading ? "Agent运行中…" : "发送给Agent" }}</button></div></form>
      </div>
      <div class="agent-result">
        <RequestState :loading="loading" :error="errorMessage" :empty="!result && !loading && !errorMessage" empty-text="运行后展示规划理由、工具调用、规范化参数与执行结果。" />
        <template v-if="result">
          <div :class="['decision-card', result.planning_status === 'tool_selected' ? 'success' : 'warning']"><span>PLANNER DECISION</span><strong>{{ result.planning_status === "tool_selected" ? "选择工具" : "拒绝请求" }}</strong><p>{{ result.plan_reason }}</p></div>
          <div class="agent-timeline">
            <article class="timeline-step complete"><span>01</span><div><strong>理解用户请求</strong><p>{{ result.user_query }}</p></div></article>
            <article class="timeline-step complete"><span>02</span><div><strong>Planner决策</strong><p>{{ result.planner_identity }}</p></div></article>
            <article :class="['timeline-step', { complete: result.tool_call }]"><span>03</span><div><strong>工具调用</strong><JsonPanel v-if="result.tool_call" :value="result.tool_call" title="查看工具与规范化参数" /><p v-else>未选择工具</p></div></article>
            <article :class="['timeline-step', { complete: result.tool_result, failed: result.tool_error }]"><span>04</span><div><strong>执行结果</strong><JsonPanel v-if="result.tool_result" :value="result.tool_result" title="查看岗位查询结果" /><JsonPanel v-else-if="result.tool_error" :value="result.tool_error" title="查看稳定错误" /><p v-else>没有执行结果</p></div></article>
          </div>
        </template>
      </div>
    </div>
  </section>
</template>
