import { requestUrl } from "obsidian";
import type { AddResult, EnrichedResult, LatticeInfo, QueryResult, RemoveResult } from "./types";

export class LatticeClient {
	constructor(private baseUrl: string) {}

	setBaseUrl(url: string): void {
		this.baseUrl = url;
	}

	async health(): Promise<boolean> {
		try {
			const res = await requestUrl({
				url: `${this.baseUrl}/health`,
				method: "GET",
				headers: { Accept: "application/json" },
			});
			return res.status === 200;
		} catch {
			return false;
		}
	}

	async info(): Promise<LatticeInfo | null> {
		try {
			const res = await requestUrl({
				url: `${this.baseUrl}/info`,
				method: "GET",
				headers: { Accept: "application/json" },
			});
			return res.json as LatticeInfo;
		} catch {
			return null;
		}
	}

	async search(
		text: string,
		topK = 10,
		enableCascade = true,
		cascadeDepth = 2,
		enableContradictions = false,
	): Promise<EnrichedResult | null> {
		try {
			const res = await requestUrl({
				url: `${this.baseUrl}/search`,
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({
					text,
					top_k: topK,
					enable_cascade: enableCascade,
					cascade_depth: cascadeDepth,
					enable_contradictions: enableContradictions,
				}),
			});
			return res.json as EnrichedResult;
		} catch {
			return null;
		}
	}

	async query(text: string, topK = 10): Promise<QueryResult | null> {
		try {
			const res = await requestUrl({
				url: `${this.baseUrl}/query`,
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({ text, top_k: topK }),
			});
			return res.json as QueryResult;
		} catch {
			return null;
		}
	}

	async add(
		text: string,
		sourceId = "",
		salience = 1.0,
		metadata?: Record<string, unknown>,
	): Promise<AddResult | null> {
		try {
			const res = await requestUrl({
				url: `${this.baseUrl}/add`,
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({
					source_id: sourceId,
					text,
					salience,
					metadata,
				}),
			});
			return res.json as AddResult;
		} catch {
			return null;
		}
	}

	async remove(sourceId: string): Promise<RemoveResult | null> {
		try {
			const res = await requestUrl({
				url: `${this.baseUrl}/remove`,
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({ source_id: sourceId }),
			});
			return res.json as RemoveResult;
		} catch {
			return null;
		}
	}
}
