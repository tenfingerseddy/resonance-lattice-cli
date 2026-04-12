import { App, Modal, setIcon } from "obsidian";
import type { LatticeClient } from "./client";
import type { ContradictionPair, EnrichedResult, SearchResult } from "./types";
import type { ResonanceLatticeSettings } from "./settings";

export class SearchModal extends Modal {
	private inputEl!: HTMLInputElement;
	private resultsContainer!: HTMLElement;
	private coverageSection!: HTMLElement;
	private contradictionsSection!: HTMLElement;
	private relatedContainer!: HTMLElement;
	private statusEl!: HTMLElement;
	private selectedIndex = -1;
	private resultEls: HTMLElement[] = [];
	private debounceTimer: ReturnType<typeof setTimeout> | null = null;
	private lastResult: EnrichedResult | null = null;

	constructor(
		app: App,
		private client: LatticeClient,
		private settings: ResonanceLatticeSettings,
	) {
		super(app);
	}

	onOpen(): void {
		const { contentEl } = this;
		contentEl.addClass("rl-search-modal");

		// Search input
		const inputContainer = contentEl.createDiv("rl-search-input-container");
		const iconEl = inputContainer.createSpan("rl-search-icon");
		setIcon(iconEl, "search");
		this.inputEl = inputContainer.createEl("input", {
			type: "text",
			placeholder: "Search your vault semantically\u2026",
			cls: "rl-search-input",
		});
		this.inputEl.focus();

		// Coverage section (confidence + per-band energy)
		this.coverageSection = contentEl.createDiv("rl-coverage-section");
		this.coverageSection.style.display = "none";

		// Status line
		this.statusEl = contentEl.createDiv("rl-status");

		// Results
		this.resultsContainer = contentEl.createDiv("rl-results");

		// Contradictions
		this.contradictionsSection = contentEl.createDiv("rl-contradictions-section");
		this.contradictionsSection.style.display = "none";

		// Related topics
		this.relatedContainer = contentEl.createDiv("rl-related");
		this.relatedContainer.style.display = "none";

		// Events
		this.inputEl.addEventListener("input", () => this.onInputChange());
		this.inputEl.addEventListener("keydown", (e) => this.onKeyDown(e));
	}

	onClose(): void {
		if (this.debounceTimer) {
			clearTimeout(this.debounceTimer);
		}
		this.contentEl.empty();
	}

	private onInputChange(): void {
		if (this.debounceTimer) {
			clearTimeout(this.debounceTimer);
		}
		const query = this.inputEl.value.trim();
		if (!query) {
			this.showEmpty();
			return;
		}
		this.debounceTimer = setTimeout(() => this.executeSearch(query), this.settings.debounceMs);
	}

	private async executeSearch(query: string): Promise<void> {
		this.statusEl.setText("Searching\u2026");
		this.statusEl.removeClass("rl-status-error");

		const result = await this.client.search(
			query,
			this.settings.topK,
			this.settings.enableCascade,
			this.settings.cascadeDepth,
			this.settings.enableContradictions,
		);

		// Stale check: only render if input hasn't changed
		if (this.inputEl.value.trim() !== query) return;

		if (!result) {
			this.showError();
			return;
		}

		this.lastResult = result;
		this.renderResults(result);
	}

	private showEmpty(): void {
		this.resultsContainer.empty();
		this.coverageSection.style.display = "none";
		this.contradictionsSection.style.display = "none";
		this.relatedContainer.style.display = "none";
		this.statusEl.setText("Type to search your vault semantically");
		this.statusEl.removeClass("rl-status-error");
		this.resultEls = [];
		this.selectedIndex = -1;
	}

	private showError(): void {
		this.resultsContainer.empty();
		this.coverageSection.style.display = "none";
		this.contradictionsSection.style.display = "none";
		this.relatedContainer.style.display = "none";
		this.statusEl.setText(
			"Server not running \u2014 run \u2018Resonance Lattice: Build & Start\u2019 from Command Palette",
		);
		this.statusEl.addClass("rl-status-error");
		this.resultEls = [];
		this.selectedIndex = -1;
	}

	private renderResults(result: EnrichedResult): void {
		this.resultsContainer.empty();
		this.resultEls = [];
		this.selectedIndex = -1;

		// ── Coverage section: confidence bar + per-band energy breakdown ──
		this.renderCoverage(result);

		// ── Status line ──
		this.statusEl.removeClass("rl-status-error");
		const count = result.results.length;
		this.statusEl.setText(
			count > 0 ? `${count} result${count !== 1 ? "s" : ""} \u00b7 ${Math.round(result.latency_ms)}ms` : "No results found",
		);

		// ── Result items ──
		for (const item of result.results) {
			const el = this.createResultItem(item, result.coverage?.band_names);
			this.resultsContainer.appendChild(el);
			this.resultEls.push(el);
		}

		// ── Contradictions ──
		this.renderContradictions(result.contradictions);

		// ── Related topics ──
		this.renderRelated(result);
	}

	// ── Coverage: overall confidence + per-band energy bars ──────────

	private renderCoverage(result: EnrichedResult): void {
		const cov = result.coverage;
		if (!cov) {
			this.coverageSection.style.display = "none";
			return;
		}

		this.coverageSection.style.display = "";
		this.coverageSection.empty();

		// Top row: confidence bar + percentage + latency
		const topRow = this.coverageSection.createDiv("rl-coverage-top");
		const pct = Math.round(cov.confidence * 100);

		const barOuter = topRow.createDiv("rl-confidence-bar");
		const barInner = barOuter.createDiv("rl-confidence-fill");
		barInner.style.width = `${pct}%`;
		barInner.addClass(
			cov.confidence >= 0.7 ? "rl-confidence-high" :
			cov.confidence >= 0.4 ? "rl-confidence-mid" : "rl-confidence-low",
		);

		topRow.createSpan("rl-confidence-label").setText(`${pct}% confidence`);
		topRow.createSpan("rl-latency").setText(`${Math.round(result.latency_ms)}ms`);

		// Per-band energy breakdown
		if (cov.band_energies && cov.band_names) {
			const maxEnergy = Math.max(...cov.band_energies, 0.001);
			const bandRow = this.coverageSection.createDiv("rl-band-breakdown");

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

		// Knowledge gap warning
		if (cov.gaps.length > 0) {
			const gapEl = this.coverageSection.createDiv("rl-gaps");
			gapEl.setText(`Low coverage: ${cov.gaps.join(", ")}`);
		}
	}

	// ── Single result item ──────────────────────────────────────────

	private createResultItem(result: SearchResult, bandNames?: string[]): HTMLElement {
		const el = createDiv("rl-result-item");

		// Header: score bar + percentage + file name + provenance badge
		const header = el.createDiv("rl-result-header");
		const scorePct = Math.round(result.score * 100);

		const scoreBar = header.createDiv("rl-score-bar");
		scoreBar.createDiv("rl-score-fill").style.width = `${scorePct}%`;

		header.createSpan("rl-score-label").setText(`${scorePct}%`);

		// File name — use top-level source_file from to_dict()
		const filePath = result.source_file || "";
		header.createSpan("rl-result-file").setText(
			filePath ? this.displayPath(filePath) : result.source_id,
		);

		// Provenance badge (dense, lexical, dense+lexical)
		if (result.provenance && result.provenance !== "dense") {
			const badge = header.createSpan("rl-provenance-badge");
			badge.setText(result.provenance);
		}

		// Heading — use top-level heading from to_dict()
		if (result.heading) {
			el.createDiv("rl-result-heading").setText(result.heading);
		}

		// Per-band score sparkline (mini bars showing which bands matched)
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

		// Preview snippet
		const text = result.summary || result.full_text || "";
		if (text) {
			el.createDiv("rl-result-preview").setText(this.truncate(text, 200));
		}

		// Click to navigate
		el.addEventListener("click", () => this.navigateToResult(result));

		return el;
	}

	// ── Contradictions ──────────────────────────────────────────────

	private renderContradictions(contradictions: ContradictionPair[]): void {
		if (!contradictions || contradictions.length === 0) {
			this.contradictionsSection.style.display = "none";
			return;
		}

		this.contradictionsSection.style.display = "";
		this.contradictionsSection.empty();
		this.contradictionsSection.createDiv("rl-section-header").setText("Contradictions");

		for (const c of contradictions) {
			const pair = this.contradictionsSection.createDiv("rl-contradiction-pair");
			const strength = Math.round(Math.abs(c.interference) * 100);

			const header = pair.createDiv("rl-contradiction-header");
			header.createSpan("rl-contradiction-strength").setText(`${strength}% conflict`);

			const sideA = pair.createDiv("rl-contradiction-side");
			sideA.createSpan("rl-contradiction-label").setText("A:");
			sideA.createSpan("rl-contradiction-text").setText(
				c.summary_a || c.source_a,
			);

			const sideB = pair.createDiv("rl-contradiction-side");
			sideB.createSpan("rl-contradiction-label").setText("B:");
			sideB.createSpan("rl-contradiction-text").setText(
				c.summary_b || c.source_b,
			);
		}
	}

	// ── Related topics ──────────────────────────────────────────────

	private renderRelated(result: EnrichedResult): void {
		if (!result.related || result.related.length === 0) {
			this.relatedContainer.style.display = "none";
			return;
		}

		this.relatedContainer.style.display = "";
		this.relatedContainer.empty();
		this.relatedContainer.createSpan("rl-related-header").setText("Related:");

		for (const topic of result.related) {
			const chip = this.relatedContainer.createSpan("rl-related-chip");
			const displayName = this.getDisplayName(topic.source_id, topic.summary);
			chip.setText(displayName);

			// Show hop depth as visual indicator
			if (topic.hop > 1) {
				const hopBadge = chip.createSpan("rl-hop-badge");
				hopBadge.setText(`${topic.hop}`);
			}

			chip.title = `${topic.summary || topic.source_id} (hop ${topic.hop}, score ${topic.score.toFixed(3)})`;
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

		// Normalize to vault-relative
		const vaultPath = filePath.replace(/^\.\//, "").replace(/^\//, "");

		// Try exact path first, then with .md extension
		let file = this.app.vault.getAbstractFileByPath(vaultPath);
		if (!file && !vaultPath.endsWith(".md")) {
			file = this.app.vault.getAbstractFileByPath(vaultPath + ".md");
		}

		if (!file) return;

		const link = result.heading ? `${file.path}#${result.heading}` : file.path;
		await this.app.workspace.openLinkText(link, "", false);
		this.close();
	}

	// ── Keyboard navigation ─────────────────────────────────────────

	private onKeyDown(e: KeyboardEvent): void {
		if (e.key === "ArrowDown") {
			e.preventDefault();
			this.moveSelection(1);
		} else if (e.key === "ArrowUp") {
			e.preventDefault();
			this.moveSelection(-1);
		} else if (e.key === "Enter") {
			e.preventDefault();
			if (this.selectedIndex >= 0 && this.lastResult) {
				this.navigateToResult(this.lastResult.results[this.selectedIndex]);
			}
		} else if (e.key === "Escape") {
			this.close();
		}
	}

	private moveSelection(delta: number): void {
		if (this.resultEls.length === 0) return;

		if (this.selectedIndex >= 0) {
			this.resultEls[this.selectedIndex].removeClass("rl-result-selected");
		}

		this.selectedIndex += delta;
		if (this.selectedIndex < 0) this.selectedIndex = this.resultEls.length - 1;
		if (this.selectedIndex >= this.resultEls.length) this.selectedIndex = 0;

		this.resultEls[this.selectedIndex].addClass("rl-result-selected");
		this.resultEls[this.selectedIndex].scrollIntoView({ block: "nearest" });
	}

	// ── Helpers ─────────────────────────────────────────────────────

	private displayPath(filePath: string): string {
		const clean = filePath.replace(/^\.\//, "");
		// Show just filename for short display, full path in title
		const parts = clean.split("/");
		if (parts.length <= 2) return clean;
		return `\u2026/${parts.slice(-2).join("/")}`;
	}

	private getDisplayName(sourceId: string, summary?: string): string {
		if (summary && summary.length <= 40) return summary;
		if (summary) return summary.slice(0, 37) + "\u2026";
		const parts = sourceId.split("/");
		const last = parts[parts.length - 1];
		return last.replace(/\.\w+$/, "").replace(/_/g, " ");
	}

	private truncate(text: string, maxLen: number): string {
		if (text.length <= maxLen) return text;
		const cut = text.lastIndexOf(" ", maxLen);
		return text.slice(0, cut > maxLen / 2 ? cut : maxLen) + "\u2026";
	}
}
