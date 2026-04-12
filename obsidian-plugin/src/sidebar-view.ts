import { ItemView, WorkspaceLeaf, setIcon } from "obsidian";
import type ResonanceLatticePlugin from "./main";
import type { ProgressEvent } from "./progress-modal";
import type { LatticeInfo } from "./types";

export const VIEW_TYPE = "resonance-lattice-sidebar";

export class ResonanceLatticeView extends ItemView {
	private statusEl!: HTMLElement;
	private infoEl!: HTMLElement;
	private progressSection!: HTMLElement;
	private barInner!: HTMLElement;
	private progressDetail!: HTMLElement;
	private progressStats!: HTMLElement;
	private cancelBtn!: HTMLElement;
	private actionsEl!: HTMLElement;

	constructor(
		leaf: WorkspaceLeaf,
		private plugin: ResonanceLatticePlugin,
	) {
		super(leaf);
	}

	getViewType(): string {
		return VIEW_TYPE;
	}

	getDisplayText(): string {
		return "Resonance Lattice";
	}

	getIcon(): string {
		return "radar";
	}

	async onOpen(): Promise<void> {
		const container = this.containerEl.children[1] as HTMLElement;
		container.empty();
		container.addClass("rl-sidebar");

		// ── Status section ──
		const statusSection = container.createDiv({ cls: "rl-sidebar-section" });
		statusSection.createEl("h4", { text: "Server" });
		this.statusEl = statusSection.createDiv({ cls: "rl-sidebar-status" });
		this.infoEl = statusSection.createDiv({ cls: "rl-sidebar-info" });

		// ── Actions ──
		this.actionsEl = container.createDiv({ cls: "rl-sidebar-section" });
		this.actionsEl.createEl("h4", { text: "Actions" });

		this.createButton(this.actionsEl, "Search", "search", () =>
			this.plugin.app.commands.executeCommandById("resonance-lattice:search"),
		);
		this.createButton(this.actionsEl, "Build", "hammer", () =>
			this.plugin.app.commands.executeCommandById("resonance-lattice:build"),
		);
		this.createButton(this.actionsEl, "Sync", "refresh-cw", () =>
			this.plugin.app.commands.executeCommandById("resonance-lattice:sync"),
		);
		this.createButton(this.actionsEl, "Build & Start", "play", () =>
			this.plugin.app.commands.executeCommandById("resonance-lattice:rebuild-and-restart"),
		);

		// ── Progress section (hidden until build starts) ──
		this.progressSection = container.createDiv({ cls: "rl-sidebar-section rl-sidebar-progress" });
		this.progressSection.createEl("h4", { text: "Progress" });
		this.progressSection.style.display = "none";

		const barOuter = this.progressSection.createDiv({ cls: "rl-progress-bar-outer" });
		this.barInner = barOuter.createDiv({ cls: "rl-progress-bar-inner" });
		this.barInner.style.width = "0%";

		this.progressDetail = this.progressSection.createDiv({ cls: "rl-progress-detail" });
		this.progressStats = this.progressSection.createDiv({ cls: "rl-progress-stats" });

		this.cancelBtn = this.progressSection.createEl("button", {
			cls: "rl-sidebar-btn rl-cancel-btn",
			text: "Cancel",
		});
		setIcon(this.cancelBtn.createSpan({ cls: "rl-btn-icon" }), "x");
		this.cancelBtn.addEventListener("click", () => this.plugin.cancelBuild());

		// Initial state
		this.refreshStatus();
	}

	async refreshStatus(): Promise<void> {
		const alive = await this.plugin.client.health();
		if (alive) {
			this.statusEl.setText("Online");
			this.statusEl.removeClass("rl-status-offline");
			this.statusEl.addClass("rl-status-online");

			const info: LatticeInfo | null = await this.plugin.client.info();
			if (info) {
				this.infoEl.empty();
				this.infoEl.createDiv({ text: `${info.source_count} sources` });
				this.infoEl.createDiv({
					text: `${info.field_type} ${info.bands}B x ${info.dim}D`,
					cls: "rl-sidebar-dim",
				});
				this.infoEl.createDiv({
					text: `${info.field_size_mb.toFixed(1)} MB` +
						(info.snr ? ` | SNR ${info.snr.toFixed(1)}` : ""),
					cls: "rl-sidebar-dim",
				});
			}
		} else {
			this.statusEl.setText("Offline");
			this.statusEl.addClass("rl-status-offline");
			this.statusEl.removeClass("rl-status-online");
			this.infoEl.empty();
		}
	}

	showProgress(): void {
		this.progressSection.style.display = "";
		this.barInner.style.width = "0%";
		this.barInner.removeClass("rl-progress-done");
		this.progressDetail.setText("Starting\u2026");
		this.progressStats.setText("");
		this.cancelBtn.style.display = "";
	}

	updateProgress(event: ProgressEvent): void {
		this.progressSection.style.display = "";

		switch (event.phase) {
			case "scanning":
				this.barInner.style.width = "5%";
				this.progressDetail.setText(
					`Found ${event.total_files} files` +
						(event.extensions ? ` (${event.extensions})` : ""),
				);
				break;

			case "encoding": {
				const pct =
					event.current && event.total
						? Math.round((event.current / event.total) * 100)
						: 0;
				this.barInner.style.width = `${pct}%`;
				this.progressDetail.setText(
					`${event.current}/${event.total}: ${event.file ?? ""}`,
				);
				const parts: string[] = [];
				if (event.status === "skipped") parts.push("skipped");
				else if (event.chunks) parts.push(`${event.chunks} chunks`);
				this.progressStats.setText(parts.join(" \u00b7 "));
				break;
			}

			case "saving":
				this.barInner.style.width = "95%";
				this.progressDetail.setText("Saving cartridge\u2026");
				break;

			case "done":
				this.barInner.style.width = "100%";
				this.barInner.addClass("rl-progress-done");
				this.cancelBtn.style.display = "none";
				if (event.chunks && event.files) {
					this.progressDetail.setText(
						`Done: ${event.chunks} chunks from ${event.files} files`,
					);
				} else {
					const parts: string[] = [];
					if (event.added) parts.push(`+${event.added}`);
					if (event.updated) parts.push(`~${event.updated}`);
					if (event.removed) parts.push(`-${event.removed}`);
					this.progressDetail.setText(
						parts.length > 0 ? `Done: ${parts.join("  ")}` : "Done",
					);
				}
				this.progressStats.setText(`${event.elapsed ?? "?"}s`);
				this.refreshStatus();
				break;
		}
	}

	hideProgress(): void {
		this.progressSection.style.display = "none";
	}

	private createButton(
		parent: HTMLElement,
		label: string,
		icon: string,
		onClick: () => void,
	): void {
		const btn = parent.createEl("button", {
			cls: "rl-sidebar-btn",
			text: label,
		});
		const iconSpan = btn.createSpan({ cls: "rl-btn-icon" });
		setIcon(iconSpan, icon);
		btn.prepend(iconSpan);
		btn.addEventListener("click", onClick);
	}

	async onClose(): Promise<void> {
		// nothing to clean up
	}
}
