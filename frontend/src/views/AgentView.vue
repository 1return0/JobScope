<script setup lang="ts">
import { onMounted, ref } from "vue";
import { ApiError, agentApi } from "../api";
import JsonPanel from "../components/JsonPanel.vue";
import RequestState from "../components/RequestState.vue";
import type { JobAgentConversationTurn, JobAgentPreference, JobAgentPreferenceKey, JobAgentResponse } from "../types";

const userQuery = ref("Find internship roles in Shanghai");
const loading = ref(false);
const errorMessage = ref("");
const result = ref<JobAgentResponse | null>(null);
const conversationTurns = ref<JobAgentConversationTurn[]>([]);
const preferences = ref<JobAgentPreference[]>([]);
const memoryError = ref("");
const memoryLoading = ref(false);
const preferenceKey = ref<JobAgentPreferenceKey>("preferred_location");
const preferenceValue = ref("");
let sessionEstablished = false;

const presets = ["Find internship roles in Shanghai", "What major requirements are stated in recruiting materials?", "Find Beijing internships and summarize education requirements from sources.", "Delete every job from the database"];

async function ensureSession() {
  if (sessionEstablished) return;
  await agentApi.establishSession();
  sessionEstablished = true;
}

async function refreshMemory() {
  memoryLoading.value = true;
  memoryError.value = "";
  try {
    const [conversation, preferenceResponse] = await Promise.all([agentApi.conversation(), agentApi.preferences()]);
    sessionEstablished = true;
    conversationTurns.value = conversation.turns;
    preferences.value = preferenceResponse.preferences;
  } catch (cause) {
    if (cause instanceof ApiError && cause.status === 401) {
      try {
        await ensureSession();
        const [conversation, preferenceResponse] = await Promise.all([agentApi.conversation(), agentApi.preferences()]);
        conversationTurns.value = conversation.turns;
        preferences.value = preferenceResponse.preferences;
      } catch (retryCause) { memoryError.value = retryCause instanceof Error ? retryCause.message : "Unable to load memory."; }
    } else memoryError.value = cause instanceof Error ? cause.message : "Unable to load memory.";
  } finally { memoryLoading.value = false; }
}

async function submit() {
  if (!userQuery.value.trim()) return;
  loading.value = true; errorMessage.value = ""; result.value = null;
  try {
    await ensureSession();
    result.value = await agentApi.query(userQuery.value.trim());
    await refreshMemory();
  } catch (error) {
    if (error instanceof ApiError && error.status === 401) sessionEstablished = false;
    if (error instanceof ApiError && error.status === 503) errorMessage.value = "Job Agent runtime is unavailable. Check the Agent switch, model configuration, and database connection.";
    else if (error instanceof ApiError && error.status === 429) errorMessage.value = "The Planner is rate-limited. Please try again later.";
    else if (error instanceof ApiError && error.status === 502) errorMessage.value = "Planner output did not pass contract validation; tool execution was blocked.";
    else errorMessage.value = error instanceof Error ? error.message : "Agent request failed.";
  } finally { loading.value = false; }
}

async function savePreference() {
  const value = preferenceValue.value.trim(); if (!value) return;
  try { await ensureSession(); await agentApi.savePreference(preferenceKey.value, value); preferenceValue.value = ""; await refreshMemory(); }
  catch (cause) { memoryError.value = cause instanceof Error ? cause.message : "Unable to save preference."; }
}
async function clearPreference(key: JobAgentPreferenceKey) {
  try { await agentApi.clearPreference(key); await refreshMemory(); }
  catch (cause) { memoryError.value = cause instanceof Error ? cause.message : "Unable to clear preference."; }
}
async function clearConversation() {
  try { await agentApi.clearConversation(); conversationTurns.value = []; }
  catch (cause) { memoryError.value = cause instanceof Error ? cause.message : "Unable to clear conversation."; }
}
onMounted(() => { void refreshMemory(); });
</script>

<template>
  <section class="page-stack">
    <div class="page-intro"><div><span class="eyebrow">LANGGRAPH · CONTROLLED TOOL USE</span><h2>Query current jobs with natural language</h2><p>The Planner can select only registered read-only tools. Pydantic validates all arguments before any database query.</p></div><span class="boundary-chip">READ-ONLY AGENT</span></div>
    <div class="agent-layout">
      <div class="agent-chat panel">
        <div class="preset-list"><button v-for="preset in presets" :key="preset" type="button" @click="userQuery = preset">{{ preset }}</button></div>
        <form class="agent-composer" @submit.prevent="submit"><textarea v-model="userQuery" maxlength="1000" rows="4" placeholder="e.g. Find Beijing internships and summarize education requirements"></textarea><div><span>Supports job filters, grounded RAG, and their combination. It never performs write operations.</span><button class="primary-button" type="submit" :disabled="loading">{{ loading ? "Running Agent…" : "Send to Agent" }}</button></div></form>
        <section class="memory-panel" aria-label="Conversation and preferences">
          <header><span>MEMORY</span><button type="button" class="text-action" :disabled="memoryLoading" @click="refreshMemory">Refresh</button></header>
          <p class="memory-hint">Short-term context is a bounded safe transcript. Preferences are saved only after this explicit action.</p>
          <div class="memory-preference-form"><select v-model="preferenceKey" aria-label="Preference key"><option value="preferred_location">Preferred location</option><option value="preferred_recruitment_type">Preferred recruitment type</option></select><input v-model="preferenceValue" maxlength="120" placeholder="e.g. Shanghai" /><button type="button" class="secondary-button" @click="savePreference">Save preference</button></div>
          <ul v-if="preferences.length" class="memory-list"><li v-for="entry in preferences" :key="entry.preference_key"><strong>{{ entry.preference_key }}</strong><span>{{ entry.preference_value }}</span><button type="button" class="text-action" @click="clearPreference(entry.preference_key)">Remove</button></li></ul>
          <div class="conversation-heading"><span>Recent conversation</span><button type="button" class="text-action" :disabled="!conversationTurns.length" @click="clearConversation">Clear</button></div>
          <ol v-if="conversationTurns.length" class="conversation-list"><li v-for="(turn, index) in conversationTurns" :key="`${index}-${turn.role}`" :class="turn.role"><strong>{{ turn.role }}</strong><span>{{ turn.content }}</span></li></ol>
          <p v-else class="memory-hint">No saved turns for this session yet.</p><p v-if="memoryError" class="memory-error">{{ memoryError }}</p>
        </section>
      </div>
      <div class="agent-result">
        <RequestState :loading="loading" :error="errorMessage" :empty="!result && !loading && !errorMessage" empty-text="Run a request to inspect planner reasoning, tool call, normalized arguments, and execution result." />
        <template v-if="result">
          <div :class="['decision-card', result.planning_status === 'tool_selected' ? 'success' : 'warning']"><span>PLANNER DECISION</span><strong>{{ result.planning_status === "tool_selected" ? "Tool selected" : "Request refused" }}</strong><p>{{ result.plan_reason }}</p></div>
          <div class="agent-timeline">
            <article class="timeline-step complete"><span>01</span><div><strong>Understand request</strong><p>{{ result.user_query }}</p></div></article>
            <article class="timeline-step complete"><span>02</span><div><strong>Planner decision</strong><p>{{ result.planner_identity }}</p></div></article>
            <article :class="['timeline-step', { complete: result.tool_call }]"><span>03</span><div><strong>Tool call</strong><JsonPanel v-if="result.tool_call" :value="result.tool_call" title="Inspect tool and normalized arguments" /><p v-else>No tool selected</p></div></article>
            <article :class="['timeline-step', { complete: result.tool_result, failed: result.tool_error }]"><span>04</span><div><strong>Execution result</strong><JsonPanel v-if="result.tool_result" :value="result.tool_result" title="Inspect job query result" /><JsonPanel v-else-if="result.tool_error" :value="result.tool_error" title="Inspect stable error" /><p v-else>No execution result</p></div></article>
          </div>
          <section v-if="result.harness" class="memory-panel" aria-label="Agent execution budget">
            <header><span>EXECUTION HARNESS</span></header>
            <p class="memory-hint">Request-scoped operational telemetry. It never includes prompts, credentials, or raw provider responses.</p>
            <ul class="memory-list">
              <li><strong>Model calls</strong><span>{{ result.harness.model_call_count }}</span></li>
              <li><strong>Tool calls</strong><span>{{ result.harness.tool_call_count }}</span></li>
              <li><strong>Elapsed</strong><span>{{ result.harness.elapsed_seconds.toFixed(3) }}s</span></li>
            </ul>
            <JsonPanel :value="result.harness.steps" title="Safe execution steps" />
          </section>
        </template>
      </div>
    </div>
  </section>
</template>
