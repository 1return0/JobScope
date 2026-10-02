import type {
  DocumentParseResponse,
  DocumentProcessResponse,
  EvaluationArtifactResponse,
  EvaluationCatalogResponse,
  GroundedAnswerResponse,
  HealthResponse,
  HybridSearchResponse,
  JobAgentResponse,
  JobAgentConversationResponse,
  JobAgentPreference,
  JobAgentPreferenceKey,
  ServiceMetaResponse,
  SourceAssessment,
  SourceCandidate,
} from "./types";

export class ApiError extends Error {
  constructor(
    message: string,
    public readonly status: number,
    public readonly detail: unknown,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(
  path: string,
  init: RequestInit = {},
): Promise<T> {
  const response = await fetch(path, {
    ...init,
    credentials: "include",
  });
  const contentType = response.headers.get("content-type") ?? "";
  const payload = contentType.includes("application/json")
    ? await response.json()
    : await response.text();
  if (!response.ok) {
    const detail = typeof payload === "object" && payload && "detail" in payload
      ? (payload as { detail: unknown }).detail
      : payload;
    const message = typeof detail === "string"
      ? detail
      : `请求失败（HTTP ${response.status}）`;
    throw new ApiError(message, response.status, detail);
  }
  return payload as T;
}

function postJson<T>(path: string, payload: unknown): Promise<T> {
  return request<T>(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

function putJson<T>(path: string, payload: unknown): Promise<T> {
  return request<T>(path, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export const systemApi = {
  health: (signal?: AbortSignal) => request<HealthResponse>("/health", { signal }),
  meta: (signal?: AbortSignal) => request<ServiceMetaResponse>("/v1/meta", { signal }),
};

export const documentApi = {
  parse(file: File, sourceReference: string) {
    return uploadDocument<DocumentParseResponse>("/v1/documents/parse", file, sourceReference);
  },
  process(file: File, sourceReference: string) {
    return uploadDocument<DocumentProcessResponse>("/v1/documents/process", file, sourceReference);
  },
};

function uploadDocument<T>(path: string, file: File, sourceReference: string): Promise<T> {
  const form = new FormData();
  form.append("file", file);
  form.append("source_reference", sourceReference);
  return request<T>(path, { method: "POST", body: form });
}

export const retrievalApi = {
  search: (query: string, topK: number) =>
    postJson<HybridSearchResponse>("/v1/retrieval/hybrid", { query, top_k: topK }),
  answer: (question: string, topK: number) =>
    postJson<GroundedAnswerResponse>("/v1/answers/grounded", { question, top_k: topK }),
};

export const agentApi = {
  establishSession: () => postJson<{ expires_at: string }>("/v1/job-agent/sessions", {}),
  query: (userQuery: string) =>
    postJson<JobAgentResponse>("/v1/job-agent/query", { user_query: userQuery }),
  conversation: () => request<JobAgentConversationResponse>("/v1/job-agent/conversation"),
  clearConversation: () => request<void>("/v1/job-agent/conversation", { method: "DELETE" }),
  preferences: () => request<{ preferences: JobAgentPreference[] }>("/v1/job-agent/memory/preferences"),
  savePreference: (preferenceKey: JobAgentPreferenceKey, preferenceValue: string) =>
    putJson<JobAgentPreference>("/v1/job-agent/memory/preferences", {
      preference_key: preferenceKey,
      preference_value: preferenceValue,
      user_consented: true,
    }),
  clearPreference: (preferenceKey: JobAgentPreferenceKey) =>
    request<void>(`/v1/job-agent/memory/preferences/${preferenceKey}`, { method: "DELETE" }),
};

export const sourceApi = {
  assess: (candidate: SourceCandidate) =>
    postJson<SourceAssessment>("/v1/source-candidates/assess", candidate),
};

export const evaluationApi = {
  catalog: () => request<EvaluationCatalogResponse>("/v1/evaluations/catalog"),
  artifact: (artifactId: string) =>
    request<EvaluationArtifactResponse>(`/v1/evaluations/artifacts/${artifactId}`),
};
