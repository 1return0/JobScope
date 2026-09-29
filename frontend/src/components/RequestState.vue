<script setup lang="ts">
defineProps<{ loading?: boolean; error?: string; empty?: boolean; emptyText?: string }>();
</script>

<template>
  <div v-if="loading" class="request-state"><span class="spinner"></span><strong>正在处理请求</strong><p>请勿重复提交，处理完成后结果会自动出现。</p></div>
  <div v-else-if="error" class="request-state request-error"><span>!</span><strong>请求未完成</strong><p>{{ error }}</p></div>
  <div v-else-if="empty" class="request-state"><span>—</span><strong>暂无结果</strong><p>{{ emptyText ?? "请先提交一次请求。" }}</p></div>
</template>
