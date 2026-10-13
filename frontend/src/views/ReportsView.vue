<script setup lang="ts">
import ReportWorkspace from "../components/ReportWorkspace.vue";
import { agentApi } from "../api";
</script>
<template><ReportWorkspace base-path="/v1/job-agent/reports" :establish-session="agentApi.establishSession" /></template>
