import { ItemView, WorkspaceLeaf, setIcon } from "obsidian";
import type ResonanceLatticePlugin from "./main";
import type { EnrichedResult, SearchResult, RelatedTopic, ContradictionPair } from "./types";
import { ResonanceGraph, buildGraphData, FULLPAGE_LAYOUT, type GraphNode } from "./resonance-graph";

export const GRAPH_VIEW_TYPE = "resonance-lattice-graph";

export class ResonanceGraphView extends ItemView {
	private inputEl!: HTMLInputElement;
	private statusEl!: HTMLElement;
	private contentArea!: HTMLElement;
	private controlPanel!: HTMLElement;
	private controlToggle!: HTMLElement;
	private xraySection!: HTMLElement;
	private graph: ResonanceGraph | null = null;
	private debounceTimer: ReturnType<typeof setTimeout> | null = null;
	private lastResult: EnrichedResult | null = null;
	private viewMode: "graph" | "list" = "graph";
	private listBtn!: HTMLElement;
	private graphBtn!: HTMLElement;

	// Advanced search options
	private enableSubgraph = false;
	private enableCrossEncoder = false;
	private boostTopics: string[] = [];
	private suppressTopics: string[] = [];

	// Display options
	private nodeScale = 1.0;
	private linkOpacity = 0.4;

	constructor(
		leaf: WorkspaceLeaf,
		private plugin: ResonanceLatticePlugin,
	) {
		super(leaf);
	}

	getViewType(): string {
		return GRAPH_VIEW_TYPE;
	}

	getDisplayText(): string {
		return "Resonance Explorer";
	}

	getIcon(): string {
		return "radar";
	}

	async onOpen(): Promise<void> {
		const container = this.containerEl.children[1] as HTMLElement;
		container.empty();
		container.addClass("rl-explorer");

		// ── Top bar ──
		const topBar = container.createDiv({ cls: "rl-explorer-topbar" });

		const searchWrap = topBar.createDiv({ cls: "rl-explorer-search" });
		const iconEl = searchWrap.createSpan({ cls: "rl-search-icon" });
		setIcon(iconEl, "search");
		this.inputEl = searchWrap.createEl("input", {
			type: "text",
			placeholder: "Search your vault semantically\u2026",
			cls: "rl-search-input",
		});
		this.inputEl.addEventListener("input", () => this.onInputChange());
		this.inputEl.addEventListener("keydown", (e) => {
			if (e.key === "Escape") this.inputEl.blur();
		});

		this.statusEl = topBar.createDiv({ cls: "rl-explorer-status" });
		this.statusEl.setText("Type to search");

		// View toggle
		const viewToggle = topBar.createDiv({ cls: "rl-view-toggle" });
		this.graphBtn = viewToggle.createEl("button", { cls: "rl-toggle-btn rl-toggle-active" });
		setIcon(this.graphBtn, "git-fork");
		this.graphBtn.title = "Graph view";
		this.graphBtn.addEventListener("click", () => this.setViewMode("graph"));

		this.listBtn = viewToggle.createEl("button", { cls: "rl-toggle-btn" });
		setIcon(this.listBtn, "list");
		this.listBtn.title = "List view";
		this.listBtn.addEventListener("click", () => this.setViewMode("list"));

		// Controls toggle
		this.controlToggle = topBar.createEl("button", { cls: "rl-toggle-btn" });
		setIcon(this.controlToggle, "sliders-horizontal");
		this.controlToggle.title = "Advanced controls";
		this.controlToggle.addEventListener("click", () => this.toggleControls());

		// ── Main body (content + optional control panel) ──
		const body = container.createDiv({ cls: "rl-explorer-body" });

		// Content area (graph or list)
		this.contentArea = body.createDiv({ cls: "rl-explorer-content" });
		const empty = this.contentArea.createDiv({ cls: "rl-fullgraph-empty" });
		empty.setText("Search to see results");

		// Control panel (hidden by default)
		this.controlPanel = body.createDiv({ cls: "rl-explorer-controls" });
		this.controlPanel.style.display = "none";
		this.buildControlPanel();

		this.inputEl.focus();
	}

	async onClose(): Promise<void> {
		if (this.debounceTimer) clearTimeout(this.debounceTimer);
		this.graph?.destroy();
		this.graph = null;
	}

	// ── Control panel ───────────────────────────────────────────────

	private buildControlPanel(): void {
		this.controlPanel.empty();

		// ── Search options ──
		const searchSection = this.controlPanel.createDiv({ cls: "rl-ctrl-section" });
		searchSection.createEl("h4", { text: "Search Options" });

		this.createToggle(searchSection, "Subgraph expansion", this.enableSubgraph, (v) => {
			this.enableSubgraph = v;
			this.rerunSearch();
		}, "Include spectral neighbors for richer context");
		this.createToggle(searchSection, "Cross-encoder rerank", this.enableCrossEncoder, (v) => {
			this.enableCrossEncoder = v;
			this.rerunSearch();
		}, "Transformer reranking \u2014 higher quality, slower");

		// ── Topic sculpting ──
		const sculptSection = this.controlPanel.createDiv({ cls: "rl-ctrl-section" });
		sculptSection.createEl("h4", { text: "Topic Sculpting" });

		const boostWrap = sculptSection.createDiv({ cls: "rl-ctrl-field" });
		boostWrap.createSpan({ cls: "rl-ctrl-label", text: "Boost" });
		const boostInput = boostWrap.createEl("input", {
			type: "text",
			placeholder: "topic1, topic2",
			cls: "rl-ctrl-input",
		});
		let boostTimer: ReturnType<typeof setTimeout> | null = null;
		boostInput.addEventListener("input", () => {
			if (boostTimer) clearTimeout(boostTimer);
			boostTimer = setTimeout(() => {
				this.boostTopics = boostInput.value.split(",").map((s) => s.trim()).filter(Boolean);
				this.rerunSearch();
			}, 800);
		});

		const suppressWrap = sculptSection.createDiv({ cls: "rl-ctrl-field" });
		suppressWrap.createSpan({ cls: "rl-ctrl-label", text: "Suppress" });
		const suppressInput = suppressWrap.createEl("input", {
			type: "text",
			placeholder: "topic1, topic2",
			cls: "rl-ctrl-input",
		});
		let suppressTimer: ReturnType<typeof setTimeout> | null = null;
		suppressInput.addEventListener("input", () => {
			if (suppressTimer) clearTimeout(suppressTimer);
			suppressTimer = setTimeout(() => {
				this.suppressTopics = suppressInput.value.split(",").map((s) => s.trim()).filter(Boolean);
				this.rerunSearch();
			}, 800);
		});

		// ── Display ──
		const displaySection = this.controlPanel.createDiv({ cls: "rl-ctrl-section" });
		displaySection.createEl("h4", { text: "Display" });

		this.createSlider(displaySection, "Similarity threshold", 0, 1, 0.05,
			this.plugin.settings.similarityThreshold ?? 0.7,
			(v) => {
				this.plugin.settings.similarityThreshold = v;
				if (this.lastResult) this.renderContent(this.lastResult);
			}, "Min cosine similarity to draw edges");

		this.createSlider(displaySection, "Node size", 0.5, 2.0, 0.1, 1.0,
			(v) => {
				this.nodeScale = v;
				if (this.lastResult) this.renderContent(this.lastResult);
			}, "Scale node radii");

		this.createSlider(displaySection, "Link opacity", 0, 1, 0.1, 0.4,
			(v) => {
				this.linkOpacity = v;
				if (this.lastResult) this.renderContent(this.lastResult);
			}, "Edge visibility");

		// ── X-ray ──
		const xrayHeader = this.controlPanel.createDiv({ cls: "rl-ctrl-section" });
		xrayHeader.createEl("h4", { text: "X-Ray Diagnostics" });
		const xrayBtn = xrayHeader.createEl("button", { cls: "rl-ctrl-btn", text: "Run X-Ray" });
		setIcon(xrayBtn.createSpan({ cls: "rl-btn-icon" }), "scan");
		xrayBtn.addEventListener("click", () => this.runXray());
		this.xraySection = xrayHeader.createDiv({ cls: "rl-xray-results" });
	}

	private createToggle(
		parent: HTMLElement,
		label: string,
		initial: boolean,
		onChange: (value: boolean) => void,
		tooltip?: string,
	): void {
		const row = parent.createDiv({ cls: "rl-ctrl-toggle" });
		const labelEl = row.createSpan({ cls: "rl-ctrl-label", text: label });
		if (tooltip) labelEl.title = tooltip;
		const toggle = row.createEl("input", { type: "checkbox" });
		toggle.checked = initial;
		toggle.addEventListener("change", () => onChange(toggle.checked));
	}

	private createSlider(
		parent: HTMLElement,
		label: string,
		min: number,
		max: number,
		step: number,
		initial: number,
		onChange: (value: number) => void,
		tooltip?: string,
	): void {
		const row = parent.createDiv({ cls: "rl-ctrl-slider" });
		const labelEl = row.createSpan({ cls: "rl-ctrl-label", text: label });
		if (tooltip) labelEl.title = tooltip;
		const slider = row.createEl("input", { type: "range" });
		slider.min = String(min);
		slider.max = String(max);
		slider.step = String(step);
		slider.value = String(initial);
		slider.addEventListener("input", () => onChange(parseFloat(slider.value)));
	}

	private toggleControls(): void {
		const visible = this.controlPanel.style.display !== "none";
		this.controlPanel.style.display = visible ? "none" : "";
		if (visible) {
			this.controlToggle.removeClass("rl-toggle-active");
		} else {
			this.controlToggle.addClass("rl-toggle-active");
		}
	}

	// ── X-ray ───────────────────────────────────────────────────────

	private async runXray(): Promise<void> {
		this.xraySection.empty();
		this.xraySection.setText("Running diagnostics\u2026");

		const result = await this.plugin.client.xray();
		if (!result) {
			this.xraySection.setText("X-ray failed \u2014 check server");
			return;
		}

		this.xraySection.empty();

		// SNR + saturation
		const summary = this.xraySection.createDiv({ cls: "rl-xray-summary" });
		summary.createSpan({ text: `SNR ${this.fmtNum(result.snr)}` });
		summary.createSpan({ text: `Saturation ${Math.round(result.saturation * 100)}%` });
		summary.createSpan({ text: `${result.source_count} sources` });

		// Per-band health — scale bar by effective_rank (more meaningful than raw SNR)
		const maxRank = Math.max(...result.band_health.map((b) => b.effective_rank), 1);
		for (const band of result.band_health) {
			const row = this.xraySection.createDiv({ cls: "rl-xray-band" });
			row.createSpan({ cls: "rl-xray-band-name", text: band.name });
			row.createSpan({
				cls: `rl-xray-label rl-xray-${band.label}`,
				text: band.label,
			});
			const barOuter = row.createDiv({ cls: "rl-xray-bar" });
			const barInner = barOuter.createDiv({ cls: "rl-xray-bar-fill" });
			const pct = Math.min(100, Math.round((band.effective_rank / maxRank) * 100));
			barInner.style.width = `${pct}%`;
			row.createSpan({ cls: "rl-xray-snr", text: `rank ${Math.round(band.effective_rank)}` });
		}

		// Diagnostics — show warnings only, skip info lines starting with *
		const warnings = (result.diagnostics || []).filter((d) =>
			d.trim().startsWith("!") || (!d.trim().startsWith("*") && d.includes("\u2014")),
		);
		if (warnings.length) {
			const diag = this.xraySection.createDiv({ cls: "rl-xray-diag" });
			for (const d of warnings) {
				const clean = d.replace(/^\s*[!*]\s*/, "");
				diag.createDiv({ text: clean, cls: "rl-xray-diag-item" });
			}
		}
	}

	// ── Search ──────────────────────────────────────────────────────

	private onInputChange(): void {
		if (this.debounceTimer) clearTimeout(this.debounceTimer);
		const query = this.inputEl.value.trim();
		if (!query) {
			this.contentArea.empty();
			const empty = this.contentArea.createDiv({ cls: "rl-fullgraph-empty" });
			empty.setText("Search to see results");
			this.statusEl.setText("Type to search");
			return;
		}
		this.debounceTimer = setTimeout(
			() => this.executeSearch(query),
			this.plugin.settings.debounceMs,
		);
	}

	private rerunSearch(): void {
		const query = this.inputEl?.value.trim();
		if (query) this.executeSearch(query);
	}

	private async executeSearch(query: string): Promise<void> {
		this.statusEl.setText("Searching\u2026");
		this.statusEl.removeClass("rl-status-error");

		const alive = await this.plugin.client.health();
		if (!alive) {
			this.statusEl.setText("Server offline \u2014 start via sidebar");
			this.statusEl.addClass("rl-status-error");
			return;
		}

		const result = await this.plugin.client.searchAdvanced(
			query,
			this.plugin.settings.topK,
			this.plugin.settings.enableCascade,
			this.plugin.settings.cascadeDepth,
			this.plugin.settings.enableContradictions,
			this.enableSubgraph,
			false, // lexical injection off — reranker handles keyword signals
			this.enableCrossEncoder,
			this.boostTopics,
			this.suppressTopics,
		);

		if (this.inputEl.value.trim() !== query) return;

		if (!result) {
			this.statusEl.setText("Search failed");
			this.statusEl.addClass("rl-status-error");
			return;
		}

		this.lastResult = result;
		const count = result.results.length;
		const conf = result.coverage ? Math.round(result.coverage.confidence * 100) : "?";
		this.statusEl.setText(
			`${count} result${count !== 1 ? "s" : ""} \u00b7 ${conf}% confidence \u00b7 ${Math.round(result.latency_ms)}ms`,
		);

		this.renderContent(result);
	}

	// ── View mode ───────────────────────────────────────────────────

	private setViewMode(mode: "graph" | "list"): void {
		this.viewMode = mode;
		if (mode === "graph") {
			this.graphBtn.addClass("rl-toggle-active");
			this.listBtn.removeClass("rl-toggle-active");
		} else {
			this.listBtn.addClass("rl-toggle-active");
			this.graphBtn.removeClass("rl-toggle-active");
		}
		this.graph?.destroy();
		this.graph = null;
		if (this.lastResult) this.renderContent(this.lastResult);
	}

	private renderContent(result: EnrichedResult): void {
		this.contentArea.empty();
		if (this.viewMode === "graph") {
			this.renderGraph(result);
		} else {
			this.renderList(result);
		}
	}

	// ── Graph view ──────────────────────────────────────────────────

	private renderGraph(result: EnrichedResult): void {
		const threshold = this.plugin.settings.similarityThreshold ?? 0.7;
		const scaledLayout = {
			...FULLPAGE_LAYOUT,
			resultMinRadius: FULLPAGE_LAYOUT.resultMinRadius * this.nodeScale,
			resultMaxRadius: FULLPAGE_LAYOUT.resultMaxRadius * this.nodeScale,
			queryRadius: FULLPAGE_LAYOUT.queryRadius * this.nodeScale,
			relatedRadius: FULLPAGE_LAYOUT.relatedRadius * this.nodeScale,
		};
		const data = buildGraphData(result, threshold, scaledLayout);

		this.graph = new ResonanceGraph(this.contentArea, scaledLayout);
		this.graph.setCallbacks(
			(node: GraphNode) => this.navigateToResult(node.result!),
			(node: GraphNode) => {
				if (node.related) {
					this.inputEl.value = node.related.summary || node.related.source_id;
					this.onInputChange();
				}
			},
		);
		this.graph.render(data);
	}

	// ── List view ───────────────────────────────────────────────────

	private renderList(result: EnrichedResult): void {
		// Coverage
		this.renderCoverage(result);

		// Results
		const resultsList = this.contentArea.createDiv({ cls: "rl-explorer-results" });
		for (const item of result.results) {
			const el = this.createResultItem(item, result.coverage?.band_names);
			resultsList.appendChild(el);
		}

		// Contradictions
		if (result.contradictions?.length) {
			this.renderContradictions(resultsList, result.contradictions);
		}

		// Related
		if (result.related?.length) {
			this.renderRelated(resultsList, result.related);
		}
	}

	private renderCoverage(result: EnrichedResult): void {
		const cov = result.coverage;
		if (!cov) return;

		const section = this.contentArea.createDiv({ cls: "rl-coverage-section" });
		const topRow = section.createDiv("rl-coverage-top");
		const pct = Math.round(cov.confidence * 100);

		const barOuter = topRow.createDiv("rl-confidence-bar");
		const barInner = barOuter.createDiv("rl-confidence-fill");
		barInner.style.width = `${pct}%`;
		barInner.addClass(
			cov.confidence >= 0.7 ? "rl-confidence-high" :
			cov.confidence >= 0.4 ? "rl-confidence-mid" : "rl-confidence-low",
		);
		topRow.createSpan("rl-confidence-label").setText(`${pct}% confidence`);

		if (cov.band_energies && cov.band_names) {
			const maxEnergy = Math.max(...cov.band_energies, 0.001);
			const bandRow = section.createDiv("rl-band-breakdown");
			for (let i = 0; i < cov.band_names.length; i++) {
				const name = cov.band_names[i];
				const energy = cov.band_energies[i];
				const bandPct = Math.round((energy / maxEnergy) * 100);
				const isGap = cov.gaps.includes(name);
				const band = bandRow.createDiv("rl-band-item");
				const label = band.createSpan("rl-band-label");
				label.setText(name);
				if (isGap) label.addClass("rl-band-gap");
				const bar = band.createDiv("rl-band-bar");
				const fill = bar.createDiv("rl-band-fill");
				fill.style.width = `${bandPct}%`;
				if (isGap) fill.addClass("rl-band-fill-gap");
			}
		}

		if (cov.gaps.length > 0) {
			section.createDiv("rl-gaps").setText(`Low coverage: ${cov.gaps.join(", ")}`);
		}
	}

	private createResultItem(result: SearchResult, bandNames?: string[]): HTMLElement {
		const el = createDiv("rl-result-item");

		const header = el.createDiv("rl-result-header");
		const scorePct = Math.round(result.score * 100);
		const scoreBar = header.createDiv("rl-score-bar");
		scoreBar.createDiv("rl-score-fill").style.width = `${scorePct}%`;
		header.createSpan("rl-score-label").setText(`${scorePct}%`);

		const filePath = result.source_file || "";
		header.createSpan("rl-result-file").setText(
			filePath ? this.displayPath(filePath) : result.source_id,
		);

		if (result.provenance && result.provenance !== "dense") {
			header.createSpan("rl-provenance-badge").setText(result.provenance);
		}

		if (result.heading) {
			el.createDiv("rl-result-heading").setText(result.heading);
		}

		if (result.band_scores && bandNames) {
			const sparkline = el.createDiv("rl-band-sparkline");
			const maxScore = Math.max(...result.band_scores, 0.001);
			for (let i = 0; i < result.band_scores.length; i++) {
				const pct = Math.round((result.band_scores[i] / maxScore) * 100);
				const bar = sparkline.createDiv("rl-spark-bar");
				bar.style.height = `${Math.max(pct, 4)}%`;
				bar.title = `${bandNames[i]}: ${result.band_scores[i].toFixed(3)}`;
			}
		}

		const text = result.summary || result.full_text || "";
		if (text) {
			const preview = text.length > 300 ? text.slice(0, 297) + "\u2026" : text;
			el.createDiv("rl-result-preview").setText(preview);
		}

		el.addEventListener("click", () => this.navigateToResult(result));
		return el;
	}

	private renderContradictions(parent: HTMLElement, contradictions: ContradictionPair[]): void {
		const section = parent.createDiv("rl-contradictions-section");
		section.createDiv("rl-section-header").setText("Contradictions");
		for (const c of contradictions) {
			const pair = section.createDiv("rl-contradiction-pair");
			const strength = Math.round(Math.abs(c.interference) * 100);
			pair.createDiv("rl-contradiction-header")
				.createSpan("rl-contradiction-strength").setText(`${strength}% conflict`);
			const sideA = pair.createDiv("rl-contradiction-side");
			sideA.createSpan("rl-contradiction-label").setText("A:");
			sideA.createSpan("rl-contradiction-text").setText(c.summary_a || c.source_a);
			const sideB = pair.createDiv("rl-contradiction-side");
			sideB.createSpan("rl-contradiction-label").setText("B:");
			sideB.createSpan("rl-contradiction-text").setText(c.summary_b || c.source_b);
		}
	}

	private renderRelated(parent: HTMLElement, related: RelatedTopic[]): void {
		const section = parent.createDiv("rl-related");
		section.createSpan("rl-related-header").setText("Related:");
		for (const topic of related) {
			const chip = section.createSpan("rl-related-chip");
			const name = topic.summary || topic.source_id;
			chip.setText(name.length > 30 ? name.slice(0, 28) + "\u2026" : name);
			if (topic.hop > 1) {
				chip.createSpan("rl-hop-badge").setText(`${topic.hop}`);
			}
			chip.title = `${name} (hop ${topic.hop}, score ${topic.score.toFixed(3)})`;
			chip.addEventListener("click", () => {
				this.inputEl.value = topic.summary || topic.source_id;
				this.onInputChange();
			});
		}
	}

	// ── Navigation ──────────────────────────────────────────────────

	private async navigateToResult(result: SearchResult): Promise<void> {
		const filePath = result.source_file;
		if (!filePath) return;

		const vaultBasePath = ((this.app.vault.adapter as any).getBasePath?.() || "") as string;
		let vaultPath = filePath;

		if (vaultBasePath) {
			const normalBase = vaultBasePath.replace(/\\/g, "/").replace(/\/$/, "");
			const normalFile = vaultPath.replace(/\\/g, "/");
			if (normalFile.startsWith(normalBase + "/")) {
				vaultPath = normalFile.slice(normalBase.length + 1);
			} else if (normalFile.toLowerCase().startsWith(normalBase.toLowerCase() + "/")) {
				vaultPath = normalFile.slice(normalBase.length + 1);
			}
		}

		vaultPath = vaultPath.replace(/^\.\//, "").replace(/^\//, "");
		let file = this.app.vault.getAbstractFileByPath(vaultPath);
		if (!file && !vaultPath.endsWith(".md")) {
			file = this.app.vault.getAbstractFileByPath(vaultPath + ".md");
		}
		if (!file) return;

		const heading = result.heading;
		const link = heading ? `${file.path}#${heading}` : file.path;
		await this.app.workspace.openLinkText(link, "", false);
	}

	private fmtNum(n: number): string {
		if (n >= 1e12) return `${(n / 1e12).toFixed(1)}T`;
		if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B`;
		if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
		if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
		return n.toFixed(1);
	}

	private displayPath(filePath: string): string {
		const clean = filePath.replace(/\\/g, "/").replace(/^\.\//, "");
		const parts = clean.split("/");
		if (parts.length <= 2) return clean;
		return `\u2026/${parts.slice(-2).join("/")}`;
	}
}
