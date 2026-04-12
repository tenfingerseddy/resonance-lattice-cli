import { Modal, App } from "obsidian";

export interface ProgressEvent {
	phase: string;
	current?: number;
	total?: number;
	file?: string;
	status?: string;
	chunks?: number;
	total_files?: number;
	extensions?: string;
	added?: number;
	updated?: number;
	removed?: number;
	skipped?: number;
	sources?: number;
	elapsed?: number;
	field_size_mb?: number;
	files?: number;
}

export class ProgressModal extends Modal {
	private headerEl!: HTMLElement;
	private barOuter!: HTMLElement;
	private barInner!: HTMLElement;
	private detailEl!: HTMLElement;
	private statsEl!: HTMLElement;
	private startTime: number;

	constructor(app: App, private title: string) {
		super(app);
		this.startTime = Date.now();
	}

	onOpen(): void {
		const { contentEl } = this;
		contentEl.empty();
		contentEl.addClass("rl-progress-modal");

		this.headerEl = contentEl.createEl("h3", { text: this.title });

		this.barOuter = contentEl.createDiv({ cls: "rl-progress-bar-outer" });
		this.barInner = this.barOuter.createDiv({ cls: "rl-progress-bar-inner" });
		this.barInner.style.width = "0%";

		this.detailEl = contentEl.createDiv({ cls: "rl-progress-detail" });
		this.detailEl.setText("Scanning files\u2026");

		this.statsEl = contentEl.createDiv({ cls: "rl-progress-stats" });
	}

	update(event: ProgressEvent): void {
		if (!this.contentEl) return;

		const elapsed = ((Date.now() - this.startTime) / 1000).toFixed(0);

		switch (event.phase) {
			case "scanning":
				this.detailEl.setText(
					`Found ${event.total_files} files` +
						(event.extensions ? ` (${event.extensions})` : ""),
				);
				this.barInner.style.width = "5%";
				break;

			case "encoding": {
				const pct =
					event.current && event.total
						? Math.round((event.current / event.total) * 100)
						: 0;
				this.barInner.style.width = `${pct}%`;
				this.detailEl.setText(
					`${event.current}/${event.total}: ${event.file ?? ""}`,
				);
				const statusParts: string[] = [];
				if (event.status === "skipped") {
					statusParts.push("skipped (unchanged)");
				} else {
					statusParts.push(
						event.status === "updated" ? "updated" : "encoded",
					);
					if (event.chunks) statusParts.push(`${event.chunks} chunks`);
				}
				this.statsEl.setText(`${statusParts.join(" \u00b7 ")}  |  ${elapsed}s`);
				break;
			}

			case "saving":
				this.barInner.style.width = "95%";
				this.detailEl.setText("Saving cartridge\u2026");
				break;

			case "done":
				this.barInner.style.width = "100%";
				this.barInner.addClass("rl-progress-done");
				this.headerEl.setText("Build complete");
				if (event.chunks && event.files) {
					this.detailEl.setText(
						`${event.chunks} chunks from ${event.files} files`,
					);
				} else if (event.sources !== undefined) {
					const parts: string[] = [];
					if (event.added) parts.push(`+${event.added} added`);
					if (event.updated) parts.push(`~${event.updated} updated`);
					if (event.removed) parts.push(`-${event.removed} removed`);
					if (event.skipped) parts.push(`${event.skipped} unchanged`);
					this.detailEl.setText(
						parts.length > 0
							? parts.join("  ")
							: "Nothing changed",
					);
				}
				this.statsEl.setText(
					`${event.elapsed ?? elapsed}s` +
						(event.field_size_mb
							? `  |  ${event.field_size_mb} MB`
							: ""),
				);
				// Auto-close after 2 seconds
				setTimeout(() => this.close(), 2000);
				break;
		}
	}

	onClose(): void {
		this.contentEl.empty();
	}
}
