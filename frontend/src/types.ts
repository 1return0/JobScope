export interface HealthResponse {
  status: string;
  service: string;
  version: string;
  environment: string;
}

export interface ServiceMetaResponse {
  stage: string;
  domain: string;
  supported_formats: string[];
  implemented_capabilities: string[];
  planned_capabilities: string[];
  read_only: boolean;
}

export interface EvidenceLocation {
  page_number: number | null;
  heading_path: string[];
  sheet_name: string | null;
  cell_reference: string | null;
  slide_number: number | null;
  table_number: number | null;
  table_row_number: number | null;
}

export interface ParsingFailure {
  code: string;
  message: string;
  retryable: boolean;
  operator_action: string;
}

export interface ParsedFragment {
  ordinal: number;
  text: string;
  location: EvidenceLocation;
}

export interface EvidenceChunk {
  evidence_id: string;
  source_fragment_ordinal: number;
  chunk_ordinal: number;
  chunker_version: string;
  text: string;
  location: EvidenceLocation;
}

export interface DocumentParseResponse {
  status: "succeeded" | "failed";
  filename: string;
  document_format: string;
  content_sha256: string;
  byte_size: number;
  fragments: ParsedFragment[];
  failure: ParsingFailure | null;
}

export interface DocumentProcessResponse {
  status: "succeeded" | "failed";
  filename: string;
  document_format: string;
  content_sha256: string;
  byte_size: number;
  fragment_count: number;
  chunk_count: number;
  chunks: EvidenceChunk[];
  failure: ParsingFailure | null;
}

export interface RetrieverRank {
  retriever: string;
  rank: number;
}

export interface HybridEvidence {
  rank: number;
  evidence_id: string;
  fusion_score: number;
  ranks_by_retriever: RetrieverRank[];
  source_reference: string;
  text: string;
  heading_path: string[];
  page_number: number | null;
}

export interface HybridSearchResponse {
  query: string;
  top_k: number;
  candidate_k: number;
  rank_constant: number;
  corpus_size: number;
  chunker_version: string;
  tokenizer_version: string;
  embedding_identity: string;
  corpus_fingerprint: string;
  results: HybridEvidence[];
}

export interface AnswerCitation {
  evidence_id: string;
  quoted_text: string;
  source_reference: string;
  heading_path: string[];
  page_number: number | null;
}

export interface AnswerClaim {
  text: string;
  citations: AnswerCitation[];
}

export interface GroundedAnswerResponse {
  question: string;
  status: "answered" | "insufficient_evidence";
  claims: AnswerClaim[];
  fallback_message: string | null;
  model_called: boolean;
  grounding_valid: boolean;
  retrieval: {
    top_k: number;
    corpus_size: number;
    corpus_fingerprint: string;
    embedding_identity: string;
  };
}

export interface JobAgentResponse {
  user_query: string;
  planning_status: "tool_selected" | "refused";
  planner_identity: string;
  plan_reason: string;
  tool_call: Record<string, unknown> | null;
  tool_result: Record<string, unknown> | null;
  tool_error: Record<string, unknown> | null;
}

export type SourceKind = "official_company" | "official_university" | "job_board";

export interface SourceCandidate {
  company: string;
  job_title: string;
  source_url: string;
  source_kind: SourceKind;
}

export interface SourceAssessment {
  source_kind: SourceKind;
  domain_verified: boolean;
  transport_secure: boolean;
  content_freshness_verified: boolean;
  trust_verified: boolean;
  manual_review_required: boolean;
  review_reasons: string[];
  matched_official_domain: string | null;
  domain_evidence_reference: string | null;
  domain_verified_at: string | null;
}

export interface EvaluationArtifactSummary {
  artifact_id: string;
  artifact_type: "dataset" | "experiment";
  family: string;
  filename: string;
  relative_path: string;
  byte_size: number;
  content_sha256: string;
  modified_at: string;
  dataset_id: string | null;
  dataset_version: number | null;
  review_status: string | null;
  case_count: number | null;
  executed_at: string | null;
  metrics: Record<string, number>;
}

export interface EvaluationCatalogResponse {
  generated_at: string;
  artifacts: EvaluationArtifactSummary[];
}

export interface EvaluationArtifactResponse {
  summary: EvaluationArtifactSummary;
  payload: Record<string, unknown> | unknown[];
}
