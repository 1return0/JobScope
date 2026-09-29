<script setup lang="ts">
import { computed } from "vue";
const props = defineProps<{ value: unknown; title?: string }>();
const formatted = computed(() => JSON.stringify(props.value, null, 2));
</script>

<template>
  <details class="json-panel">
    <summary>{{ title ?? "查看原始响应" }}</summary>
    <pre>{{ formatted }}</pre>
  </details>
</template>
