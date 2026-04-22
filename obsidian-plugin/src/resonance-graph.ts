import type { EnrichedResult, SearchResult, RelatedTopic } from "./types";

// ── Data model ──────────────────────────────────────────────────────

export interface GraphNode {
	id: string;
	type: "query" | "result" | "related";
	x: number;
	y: number;
	radius: number;
	bandScores?: number[];
	score: number;
	hop?: number;
	label: string;
	tooltip: string;
	result?: SearchResult;
	related?: RelatedTopic;
}

export interface GraphEdge {
	source: string;
	target: string;
	type: "similarity" | "contradiction";
	weight: number;
	label?: string;
}

export interface GraphData {
	nodes: GraphNode[];
	edges: GraphEdge[];
	bandNames: string[];
	confidence: number;
}

// ── Band colors ─────────────────────────────────────────────────────

const BAND_COLORS = [
	"var(--text-accent, #7c3aed)",       // domain — purple
	"var(--interactive-accent, #3b82f6)", // topic — blue
	"hsl(150, 60%, 45%)",                // relations — green
	"hsl(35, 85%, 50%)",                 // entity — amber
	"hsl(340, 65%, 50%)",                // verbatim — rose
];

// ── Math utilities ──────────────────────────────────────────────────

function cosineSimilarity(a: number[], b: number[]): number {
	let dot = 0, magA = 0, magB = 0;
	for (let i = 0; i < a.length; i++) {
		dot += a[i] * b[i];
		magA += a[i] * a[i];
		magB += b[i] * b[i];
	}
	const denom = Math.sqrt(magA) * Math.sqrt(magB);
	return denom > 0 ? dot / denom : 0;
}

function argmax(arr: number[]): number {
	let best = 0;
	for (let i = 1; i < arr.length; i++) {
		if (arr[i] > arr[best]) best = i;
	}
	return best;
}

// ── Layout presets ──────────────────────────────────────────────────

export interface LayoutConfig {
	width: number;
	height: number;
	layoutRadius: number;
	queryRadius: number;
	resultMinRadius: number;
	resultMaxRadius: number;
	relatedRadius: number;
	showLabels: boolean;
	repelStrength: number; // collision resolution iterations (0 = off, higher = more spread)
}

export const MODAL_LAYOUT: LayoutConfig = {
	width: 600, height: 400, layoutRadius: 170,
	queryRadius: 18, resultMinRadius: 5, resultMaxRadius: 16, relatedRadius: 7,
	showLabels: false, repelStrength: 2,
};

export const FULLPAGE_LAYOUT: LayoutConfig = {
	width: 1400, height: 1000, layoutRadius: 400,
	queryRadius: 28, resultMinRadius: 10, resultMaxRadius: 28, relatedRadius: 12,
	showLabels: true, repelStrength: 5,
};

// ── Build graph data from search results ────────────────────────────

export function buildGraphData(
	result: EnrichedResult,
	similarityThreshold: number,
	layout: LayoutConfig = MODAL_LAYOUT,
): GraphData {
	const bandNames = result.coverage?.band_names ?? [];
	const numBands = Math.max(bandNames.length, 1);
	const CX = layout.width / 2;
	const CY = layout.height / 2;

	const nodes: GraphNode[] = [];
	const edges: GraphEdge[] = [];

	// Query node
	const queryBandScores = result.coverage?.band_energies ?? [];
	nodes.push({
		id: "__query__",
		type: "query",
		x: CX,
		y: CY,
		radius: layout.queryRadius,
		bandScores: queryBandScores,
		score: result.coverage?.confidence ?? 1,
		label: result.query,
		tooltip: `Query: ${result.query}\nConfidence: ${Math.round((result.coverage?.confidence ?? 0) * 100)}%`,
	});

	// Dedup: keep only the highest-scoring chunk per source_file
	const seenFiles = new Map<string, SearchResult>();
	for (const r of result.results) {
		const key = r.source_file || r.source_id;
		const existing = seenFiles.get(key);
		if (!existing || r.score > existing.score) {
			seenFiles.set(key, r);
		}
	}
	const dedupedResults = [...seenFiles.values()];

	// Sort by dominant band (for grouping), then by score within band
	const sorted = dedupedResults.map((r, i) => ({
		idx: i,
		result: r,
		band: r.band_scores?.length ? argmax(r.band_scores) : 0,
	}));
	sorted.sort((a, b) => a.band !== b.band ? a.band - b.band : b.result.score - a.result.score);

	// Place results across full 360 degrees, band-grouped but evenly spread
	const n = sorted.length;
	for (let j = 0; j < n; j++) {
		const { result: r } = sorted[j];
		// Even angular distribution across full circle with band grouping preserved
		const angle = -Math.PI / 2 + (2 * Math.PI * j) / Math.max(n, 1);
		// Radial distance: high score = close to center, low score = outer edge
		// Add jitter so same-score results don't stack
		const jitter = ((j * 7919) % 100) / 100 * 0.15; // deterministic pseudo-random
		const dist = layout.layoutRadius * (0.20 + 0.55 * (1 - r.score) + jitter * 0.10);
		const radius = layout.resultMinRadius + (layout.resultMaxRadius - layout.resultMinRadius) * r.score;

			const file = r.source_file ?? r.source_id;
			const parts = file.split(/[/\\]/);
			const shortFile = parts.length > 2
				? `…/${parts.slice(-2).join("/")}`
				: file;

			const topBands: string[] = [];
			if (r.band_scores?.length && bandNames.length) {
				const indexed = r.band_scores.map((s, i) => ({ s, i }));
				indexed.sort((a, b) => b.s - a.s);
				for (let k = 0; k < Math.min(2, indexed.length); k++) {
					if (indexed[k].s > 0) {
						topBands.push(`${bandNames[indexed[k].i]} ${Math.round(indexed[k].s * 100)}%`);
					}
				}
			}

			nodes.push({
				id: r.source_id,
				type: "result",
				x: CX + dist * Math.cos(angle),
				y: CY + dist * Math.sin(angle),
				radius,
				bandScores: r.band_scores ?? [],
				score: r.score,
				label: shortFile,
				tooltip: [
					shortFile,
					r.heading ? `§ ${r.heading}` : "",
					`Score: ${Math.round(r.score * 100)}%`,
					topBands.length ? topBands.join(" | ") : "",
					r.provenance && r.provenance !== "dense" ? `(${r.provenance})` : "",
					r.summary ? r.summary.slice(0, 120) : "",
				].filter(Boolean).join("\n"),
				result: r,
		});
	}

	// Related topic nodes — outer ring
	const relatedRing = layout.layoutRadius * 0.95;
	for (let i = 0; i < result.related.length; i++) {
		const rel = result.related[i];
		const angle = -Math.PI / 2 + (2 * Math.PI * i) / Math.max(result.related.length, 1);
		nodes.push({
			id: `__related_${i}`,
			type: "related",
			x: CX + relatedRing * Math.cos(angle),
			y: CY + relatedRing * Math.sin(angle),
			radius: layout.relatedRadius,
			score: rel.score,
			hop: rel.hop,
			label: rel.summary ?? rel.source_id,
			tooltip: `${rel.summary ?? rel.source_id}\nHop: ${rel.hop} | Score: ${Math.round(rel.score * 100)}%`,
			related: rel,
		});
	}

	// Collision avoidance — multiple passes for better spread
	for (let pass = 0; pass < layout.repelStrength; pass++) {
		resolveCollisions(nodes, layout);
	}

	// Inter-result similarity edges
	const resultNodes = nodes.filter((n) => n.type === "result");
	for (let i = 0; i < resultNodes.length; i++) {
		for (let j = i + 1; j < resultNodes.length; j++) {
			const a = resultNodes[i];
			const b = resultNodes[j];
			if (a.bandScores?.length && b.bandScores?.length) {
				const sim = cosineSimilarity(a.bandScores, b.bandScores);
				if (sim >= similarityThreshold) {
					edges.push({
						source: a.id,
						target: b.id,
						type: "similarity",
						weight: sim,
					});
				}
			}
		}
	}

	// Contradiction edges
	for (const c of result.contradictions) {
		const sourceNode = nodes.find((n) => n.id === c.source_a);
		const targetNode = nodes.find((n) => n.id === c.source_b);
		if (sourceNode && targetNode) {
			edges.push({
				source: c.source_a,
				target: c.source_b,
				type: "contradiction",
				weight: Math.abs(c.interference),
				label: `${Math.round(Math.abs(c.interference) * 100)}%`,
			});
		}
	}

	return {
		nodes,
		edges,
		bandNames,
		confidence: result.coverage?.confidence ?? 0,
	};
}

function resolveCollisions(nodes: GraphNode[], layout: LayoutConfig): void {
	const minGap = layout.showLabels ? 18 : 6; // more gap when labels are visible
	const CX = layout.width / 2;
	const CY = layout.height / 2;
	for (let i = 0; i < nodes.length; i++) {
		for (let j = i + 1; j < nodes.length; j++) {
			const a = nodes[i];
			const b = nodes[j];
			const dx = b.x - a.x;
			const dy = b.y - a.y;
			const dist = Math.sqrt(dx * dx + dy * dy);
			const minDist = a.radius + b.radius + minGap;
			if (dist < minDist && dist > 0.01) {
				const push = (minDist - dist) / 2 + 2;
				const nx = dx / dist;
				const ny = dy / dist;
				if (a.type === "query") {
					b.x += nx * push * 2;
					b.y += ny * push * 2;
				} else if (b.type === "query") {
					a.x -= nx * push * 2;
					a.y -= ny * push * 2;
				} else {
					// Push both apart equally
					a.x -= nx * push;
					a.y -= ny * push;
					b.x += nx * push;
					b.y += ny * push;
				}
			}
		}
	}
	// Clamp to viewBox bounds
	const margin = 30;
	for (const n of nodes) {
		if (n.type === "query") continue;
		n.x = Math.max(margin, Math.min(layout.width - margin, n.x));
		n.y = Math.max(margin, Math.min(layout.height - margin, n.y));
	}
}

// ── SVG Renderer ────────────────────────────────────────────────────

export type NodeClickCallback = (node: GraphNode) => void;

export class ResonanceGraph {
	private svg: SVGSVGElement | null = null;
	private tooltipEl: HTMLElement | null = null;
	private container: HTMLElement;
	private data: GraphData | null = null;
	private nodeMap: Map<string, GraphNode> = new Map();
	private highlightedId: string | null = null;
	private onResultClick: NodeClickCallback | null = null;
	private onRelatedClick: NodeClickCallback | null = null;
	private layout: LayoutConfig = MODAL_LAYOUT;

	constructor(container: HTMLElement, layout: LayoutConfig = MODAL_LAYOUT) {
		this.container = container;
		this.layout = layout;
	}

	setCallbacks(
		onResultClick: NodeClickCallback,
		onRelatedClick: NodeClickCallback,
	): void {
		this.onResultClick = onResultClick;
		this.onRelatedClick = onRelatedClick;
	}

	render(data: GraphData): void {
		this.data = data;
		this.nodeMap.clear();
		for (const n of data.nodes) this.nodeMap.set(n.id, n);

		// Clear previous
		this.container.querySelectorAll(".rl-graph, .rl-graph-tooltip, .rl-graph-legend-bar").forEach((el) => el.remove());

		// Legend bar (DOM, above SVG)
		this.renderLegend(data.bandNames);

		// SVG
		const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
		svg.setAttribute("viewBox", `0 0 ${this.layout.width} ${this.layout.height}`);
		svg.setAttribute("preserveAspectRatio", "xMidYMid meet");
		svg.classList.add("rl-graph");
		this.svg = svg;

		// Edges layer
		const edgesG = this.createSvgEl("g", { class: "rl-graph-edges" });
		svg.appendChild(edgesG);
		this.renderEdges(edgesG, data);

		// Nodes layer
		const nodesG = this.createSvgEl("g", { class: "rl-graph-nodes" });
		svg.appendChild(nodesG);
		this.renderNodes(nodesG, data);

		this.container.appendChild(svg);

		// Tooltip element
		this.tooltipEl = document.createElement("div");
		this.tooltipEl.classList.add("rl-graph-tooltip");
		this.tooltipEl.style.display = "none";
		this.container.appendChild(this.tooltipEl);
	}

	destroy(): void {
		this.container.querySelectorAll(".rl-graph, .rl-graph-tooltip, .rl-graph-legend-bar").forEach((el) => el.remove());
		this.svg = null;
		this.tooltipEl = null;
		this.data = null;
		this.nodeMap.clear();
	}

	// ── Legend ───────────────────────────────────────────────────────

	private renderLegend(bandNames: string[]): void {
		const bar = document.createElement("div");
		bar.classList.add("rl-graph-legend-bar");
		for (let i = 0; i < bandNames.length; i++) {
			const item = document.createElement("span");
			item.classList.add("rl-legend-item");
			const dot = document.createElement("span");
			dot.classList.add("rl-legend-dot");
			dot.style.backgroundColor = BAND_COLORS[i % BAND_COLORS.length];
			item.appendChild(dot);
			item.appendText(bandNames[i]);
			bar.appendChild(item);
		}
		this.container.appendChild(bar);
	}

	// ── Edges ───────────────────────────────────────────────────────

	private renderEdges(parent: SVGGElement, data: GraphData): void {
		for (const edge of data.edges) {
			const source = this.nodeMap.get(edge.source);
			const target = this.nodeMap.get(edge.target);
			if (!source || !target) continue;

			const line = this.createSvgEl("line", {
				x1: String(source.x),
				y1: String(source.y),
				x2: String(target.x),
				y2: String(target.y),
				class: edge.type === "contradiction"
					? "rl-edge-contradiction"
					: "rl-edge-similarity",
				"data-source": edge.source,
				"data-target": edge.target,
			});

			if (edge.type === "similarity") {
				const width = 0.5 + 2 * edge.weight;
				const opacity = 0.1 + 0.3 * edge.weight;
				line.setAttribute("stroke-width", String(width));
				line.setAttribute("opacity", String(opacity));
			} else {
				line.setAttribute("stroke-width", "1.5");
				line.setAttribute("opacity", "0.7");
			}

			parent.appendChild(line);
		}
	}

	// ── Nodes ───────────────────────────────────────────────────────

	private renderNodes(parent: SVGGElement, data: GraphData): void {
		// Result + related first, query on top
		const sorted = [
			...data.nodes.filter((n) => n.type === "result"),
			...data.nodes.filter((n) => n.type === "related"),
			...data.nodes.filter((n) => n.type === "query"),
		];

		for (const node of sorted) {
			const g = this.createSvgEl("g", {
				class: `rl-graph-node rl-graph-${node.type}`,
				transform: `translate(${node.x}, ${node.y})`,
				"data-id": node.id,
			});

			if (node.type === "query") {
				this.renderQueryNode(g, node);
			} else if (node.type === "result") {
				this.renderResultNode(g, node);
			} else {
				this.renderRelatedNode(g, node);
			}

			// Interactions
			g.addEventListener("mouseenter", (e) => this.onNodeHover(node, e));
			g.addEventListener("mouseleave", () => this.onNodeLeave());
			g.addEventListener("click", () => this.onNodeClick(node));

			parent.appendChild(g);
		}
	}

	private renderQueryNode(g: SVGGElement, node: GraphNode): void {
		// Background circle
		const bg = this.createSvgEl("circle", {
			r: String(node.radius),
			class: "rl-node-query",
		});
		g.appendChild(bg);

		// Band ring if we have band scores
		if (node.bandScores?.length) {
			this.renderBandRing(g, node.bandScores, node.radius + 3, 3);
		}

		// "Q" label
		const text = this.createSvgEl("text", {
			class: "rl-node-label-query",
			"text-anchor": "middle",
			"dominant-baseline": "central",
			"font-size": "11",
			fill: "var(--text-on-accent, #fff)",
		});
		text.textContent = "Q";
		g.appendChild(text);
	}

	private renderResultNode(g: SVGGElement, node: GraphNode): void {
		// Background circle
		const bg = this.createSvgEl("circle", {
			r: String(node.radius),
			class: "rl-node-bg",
		});
		g.appendChild(bg);

		// Band ring
		if (node.bandScores?.length) {
			this.renderBandRing(g, node.bandScores, node.radius, 3.5);
		} else {
			const outline = this.createSvgEl("circle", {
				r: String(node.radius),
				fill: "none",
				stroke: "var(--text-muted)",
				"stroke-width": "2",
			});
			g.appendChild(outline);
		}

		// Label (full-page mode only)
		if (this.layout.showLabels) {
			const label = this.createSvgEl("text", {
				class: "rl-node-label",
				"text-anchor": "middle",
				y: String(node.radius + 14),
				"font-size": "9",
				fill: "var(--text-muted)",
			});
			// Show short filename
			const parts = node.label.split(/[/\\]/);
			const short = parts[parts.length - 1].replace(/\.\w+$/, "");
			label.textContent = short.length > 20 ? short.slice(0, 18) + "\u2026" : short;
			g.appendChild(label);
		}
	}

	private renderRelatedNode(g: SVGGElement, node: GraphNode): void {
		// Diamond (rotated square)
		const size = node.radius;
		const diamond = this.createSvgEl("rect", {
			x: String(-size),
			y: String(-size),
			width: String(size * 2),
			height: String(size * 2),
			rx: "2",
			transform: "rotate(45)",
			class: "rl-node-related",
		});
		g.appendChild(diamond);

		// Hop badge or label
		if (this.layout.showLabels) {
			const label = this.createSvgEl("text", {
				class: "rl-node-label",
				"text-anchor": "middle",
				y: String(size + 16),
				"font-size": "8",
				fill: "var(--text-faint)",
			});
			const name = node.label.length > 24 ? node.label.slice(0, 22) + "\u2026" : node.label;
			label.textContent = node.hop && node.hop > 1 ? `h${node.hop} ${name}` : name;
			g.appendChild(label);
		} else if (node.hop && node.hop > 1) {
			const badge = this.createSvgEl("text", {
				"text-anchor": "middle",
				"dominant-baseline": "central",
				"font-size": "8",
				fill: "var(--text-muted)",
				y: String(size + 10),
			});
			badge.textContent = `h${node.hop}`;
			g.appendChild(badge);
		}
	}

	// ── Band ring (donut arcs) ──────────────────────────────────────

	private renderBandRing(
		g: SVGGElement,
		scores: number[],
		radius: number,
		strokeWidth: number,
	): void {
		const total = scores.reduce((s, v) => s + Math.max(v, 0), 0);
		if (total <= 0) return;

		const circumference = 2 * Math.PI * radius;
		let cumulative = 0;

		for (let i = 0; i < scores.length; i++) {
			const fraction = Math.max(scores[i], 0) / total;
			if (fraction <= 0) continue;

			const segLen = circumference * fraction;
			const gap = circumference - segLen;

			const arc = this.createSvgEl("circle", {
				r: String(radius),
				fill: "none",
				stroke: BAND_COLORS[i % BAND_COLORS.length],
				"stroke-width": String(strokeWidth),
				"stroke-dasharray": `${segLen} ${gap}`,
				"stroke-dashoffset": String(-cumulative),
				"stroke-linecap": "butt",
			});
			g.appendChild(arc);
			cumulative += segLen;
		}
	}

	// ── Interactions ────────────────────────────────────────────────

	private onNodeHover(node: GraphNode, event: Event): void {
		this.highlightedId = node.id;
		this.showTooltip(node, event as MouseEvent);
		this.applyHighlight(node.id);
	}

	private onNodeLeave(): void {
		this.highlightedId = null;
		this.hideTooltip();
		this.clearHighlight();
	}

	private onNodeClick(node: GraphNode): void {
		if (node.type === "result" && node.result && this.onResultClick) {
			this.onResultClick(node);
		} else if (node.type === "related" && node.related && this.onRelatedClick) {
			this.onRelatedClick(node);
		}
	}

	private showTooltip(node: GraphNode, event: MouseEvent): void {
		if (!this.tooltipEl) return;
		this.tooltipEl.style.display = "";
		this.tooltipEl.innerHTML = "";

		const lines = node.tooltip.split("\n");
		for (const line of lines) {
			const div = document.createElement("div");
			div.textContent = line;
			this.tooltipEl.appendChild(div);
		}

		// Position near the mouse
		const rect = this.container.getBoundingClientRect();
		const mx = event.clientX - rect.left;
		const my = event.clientY - rect.top;
		this.tooltipEl.style.left = `${mx + 12}px`;
		this.tooltipEl.style.top = `${my - 10}px`;
	}

	private hideTooltip(): void {
		if (this.tooltipEl) {
			this.tooltipEl.style.display = "none";
		}
	}

	private applyHighlight(nodeId: string): void {
		if (!this.svg || !this.data) return;

		// Find connected node IDs
		const connected = new Set<string>([nodeId]);
		for (const edge of this.data.edges) {
			if (edge.source === nodeId) connected.add(edge.target);
			if (edge.target === nodeId) connected.add(edge.source);
		}

		// Dim the whole graph
		this.svg.classList.add("rl-graph-dimmed");

		// Un-dim connected nodes
		this.svg.querySelectorAll<SVGGElement>(".rl-graph-node").forEach((el) => {
			const id = el.getAttribute("data-id");
			if (id && connected.has(id)) {
				el.classList.add("rl-graph-highlighted");
			}
		});

		// Un-dim connected edges
		this.svg.querySelectorAll<SVGLineElement>("line").forEach((el) => {
			const src = el.getAttribute("data-source");
			const tgt = el.getAttribute("data-target");
			if ((src === nodeId || tgt === nodeId)) {
				el.classList.add("rl-edge-highlighted");
			}
		});
	}

	private clearHighlight(): void {
		if (!this.svg) return;
		this.svg.classList.remove("rl-graph-dimmed");
		this.svg.querySelectorAll(".rl-graph-highlighted").forEach((el) =>
			el.classList.remove("rl-graph-highlighted"),
		);
		this.svg.querySelectorAll(".rl-edge-highlighted").forEach((el) =>
			el.classList.remove("rl-edge-highlighted"),
		);
	}

	// ── SVG helpers ─────────────────────────────────────────────────

	private createSvgEl(tag: string, attrs: Record<string, string> = {}): SVGElement {
		const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
		for (const [k, v] of Object.entries(attrs)) {
			el.setAttribute(k, v);
		}
		return el as SVGElement;
	}
}
