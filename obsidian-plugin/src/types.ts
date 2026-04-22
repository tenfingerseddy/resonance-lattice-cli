export interface SearchResult {
	source_id: string;
	score: number;
	raw_score?: number;
	band_scores?: number[];
	summary?: string;
	full_text?: string;
	source_file?: string;
	heading?: string;
	provenance?: string;
}

export interface CoverageProfile {
	band_energies: number[];
	band_names: string[];
	total_energy: number;
	confidence: number;
	gaps: string[];
}

export interface RelatedTopic {
	source_id: string;
	score: number;
	hop: number;
	summary?: string;
}

export interface ContradictionPair {
	source_a: string;
	source_b: string;
	interference: number;
	summary_a?: string;
	summary_b?: string;
}

export interface EnrichedResult {
	query: string;
	results: SearchResult[];
	coverage: CoverageProfile;
	related: RelatedTopic[];
	contradictions: ContradictionPair[];
	latency_ms: number;
	timings_ms: Record<string, number>;
}

export interface QueryResult {
	query: string;
	latency_ms: number;
	results: {
		source_id: string;
		score: number;
		summary?: string;
		full_text?: string;
	}[];
}

export interface LatticeInfo {
	source_count: number;
	bands: number;
	dim: number;
	field_type: string;
	field_size_mb: number;
	snr?: number;
}

export interface AddResult {
	source_id: string;
	source_count: number;
}

export interface RemoveResult {
	removed: boolean;
	source_count: number;
}

export interface BandHealth {
	name: string;
	band: number;
	label: string; // "rich" | "adequate" | "thin" | "noisy"
	effective_rank: number;
	entropy: number;
	spectral_gap: number;
	snr: number;
	condition: number;
}

export interface XRayResult {
	source_count: number;
	bands: number;
	dim: number;
	snr: number;
	saturation: number;
	band_health: BandHealth[];
	diagnostics: string[];
}

export interface QueryLocation {
	query: string;
	band_energies: Record<string, number>;
	band_focus: string;
	band_focus_pct: number;
	anti_resonance_ratio: number;
	coverage: string;
	mahalanobis: number;
	uncertainty_per_band: number[];
	fisher_per_band: number[];
	expansion_hint: string | null;
}
