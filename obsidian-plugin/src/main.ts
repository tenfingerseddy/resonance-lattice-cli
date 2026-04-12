import { Notice, Plugin, TFile } from "obsidian";
import { LatticeClient } from "./client";
import { ProcessManager } from "./process";
import { ProgressModal, type ProgressEvent } from "./progress-modal";
import { SearchModal } from "./search-modal";
import { ResonanceLatticeView, VIEW_TYPE } from "./sidebar-view";
import {
	DEFAULT_SETTINGS,
	ResonanceLatticeSettingTab,
	type ResonanceLatticeSettings,
} from "./settings";
import * as path from "path";
import { existsSync } from "fs";

export default class ResonanceLatticePlugin extends Plugin {
	settings!: ResonanceLatticeSettings;
	client!: LatticeClient;
	processManager!: ProcessManager;
	private statusBarEl: HTMLElement | null = null;
	private syncTimer: ReturnType<typeof setTimeout> | null = null;
	private pendingChanges: Set<string> = new Set();
	private pendingDeletes: Set<string> = new Set();
	private sidebarView: ResonanceLatticeView | null = null;

	async onload(): Promise<void> {
		await this.loadSettings();

		this.client = new LatticeClient(`http://127.0.0.1:${this.settings.port}`);
		this.processManager = new ProcessManager(this.client);

		// ── Sidebar view ──
		this.registerView(VIEW_TYPE, (leaf) => {
			this.sidebarView = new ResonanceLatticeView(leaf, this);
			return this.sidebarView;
		});

		// Ribbon icon to open sidebar
		this.addRibbonIcon("radar", "Resonance Lattice", () => {
			this.activateSidebar();
		});

		// Status bar
		this.statusBarEl = this.addStatusBarItem();
		this.updateStatusBar("offline");

		// ── Commands ──

		this.addCommand({
			id: "search",
			name: "Search",
			hotkeys: [{ modifiers: ["Mod", "Shift"], key: "r" }],
			callback: () => this.openSearch(),
		});

		this.addCommand({
			id: "build",
			name: "Build cartridge from vault",
			callback: () => this.buildCartridge(),
		});

		this.addCommand({
			id: "sync",
			name: "Sync cartridge (incremental)",
			callback: () => this.syncCartridge(),
		});

		this.addCommand({
			id: "rebuild-and-restart",
			name: "Build & Start",
			callback: () => this.rebuildAndRestart(),
		});

		this.addCommand({
			id: "server-info",
			name: "Show server info",
			callback: () => this.showServerInfo(),
		});

		this.addCommand({
			id: "cancel-build",
			name: "Cancel build",
			callback: () => this.cancelBuild(),
		});

		this.addCommand({
			id: "open-sidebar",
			name: "Open sidebar",
			callback: () => this.activateSidebar(),
		});

		// Settings tab
		this.addSettingTab(new ResonanceLatticeSettingTab(this.app, this));

		// ── File watching for live updates ──

		// On file save: queue live update via /add endpoint
		this.registerEvent(
			this.app.vault.on("modify", (file) => {
				if (file instanceof TFile && file.extension === "md") {
					this.onFileChanged(file);
				}
			}),
		);

		// On file create: queue live update
		this.registerEvent(
			this.app.vault.on("create", (file) => {
				if (file instanceof TFile && file.extension === "md") {
					this.onFileChanged(file);
				}
			}),
		);

		// On file delete: queue live removal via /remove endpoint
		this.registerEvent(
			this.app.vault.on("delete", (file) => {
				if (file instanceof TFile && file.extension === "md") {
					this.onFileDeleted(file);
				}
			}),
		);

		// On file rename: remove old, add new
		this.registerEvent(
			this.app.vault.on("rename", (file, oldPath) => {
				if (file instanceof TFile && file.extension === "md") {
					this.pendingDeletes.add(oldPath);
					this.onFileChanged(file);
				}
			}),
		);

		// Auto-start server if cartridge exists
		this.app.workspace.onLayoutReady(() => {
			this.tryAutoStart();
		});
	}

	async onunload(): Promise<void> {
		if (this.syncTimer) {
			clearTimeout(this.syncTimer);
		}
		this.processManager.destroy();
		this.app.workspace.detachLeavesOfType(VIEW_TYPE);
	}

	async loadSettings(): Promise<void> {
		this.settings = Object.assign({}, DEFAULT_SETTINGS, await this.loadData());
	}

	async saveSettings(): Promise<void> {
		await this.saveData(this.settings);
		this.client.setBaseUrl(`http://127.0.0.1:${this.settings.port}`);
	}

	private getCartridgePath(): string {
		const adapter = this.app.vault.adapter;
		const basePath = (adapter as any).getBasePath?.() || "";
		return path.join(basePath, ".obsidian", "plugins", "resonance-lattice", "vault.rlat");
	}

	private getVaultPath(): string {
		const adapter = this.app.vault.adapter;
		return (adapter as any).getBasePath?.() || "";
	}

	// ── Search ──────────────────────────────────────────────────────

	private async openSearch(): Promise<void> {
		// Auto-restart server if it died
		const alive = await this.client.health();
		if (!alive) {
			const started = await this.startServer();
			if (!started) {
				new SearchModal(this.app, this.client, this.settings).open();
				return;
			}
		}
		new SearchModal(this.app, this.client, this.settings).open();
	}

	// ── Build & Sync ────────────────────────────────────────────────

	private onProgress(event: ProgressEvent): void {
		this.sidebarView?.updateProgress(event);
	}

	private async buildCartridge(): Promise<void> {
		const vaultPath = this.getVaultPath();
		if (!vaultPath) {
			new Notice("Could not determine vault path.");
			return;
		}

		const cartridgePath = this.getCartridgePath();
		const modal = new ProgressModal(this.app, "Building cartridge\u2026");
		modal.open();
		this.sidebarView?.showProgress();

		const result = await this.processManager.build(
			this.settings.rlatPath,
			vaultPath,
			cartridgePath,
			this.settings.encoder || undefined,
			this.settings.checkpoint || undefined,
			(event) => {
				modal.update(event);
				this.onProgress(event);
			},
		);

		if (!result.success) {
			modal.close();
			this.sidebarView?.hideProgress();
			new Notice(`Resonance Lattice: Build failed.\n${result.output}`, 10000);
		}
	}

	private async syncCartridge(): Promise<void> {
		const vaultPath = this.getVaultPath();
		const cartridgePath = this.getCartridgePath();

		if (!existsSync(cartridgePath)) {
			new Notice("No cartridge found. Run 'Build cartridge from vault' first.");
			return;
		}

		// Stop server for sync (sync modifies the .rlat file on disk)
		const wasRunning = this.processManager.isServerRunning;
		if (wasRunning) {
			this.processManager.stopServer();
			this.updateStatusBar("syncing\u2026");
		}

		const modal = new ProgressModal(this.app, "Syncing cartridge\u2026");
		modal.open();
		this.sidebarView?.showProgress();

		const result = await this.processManager.sync(
			this.settings.rlatPath,
			vaultPath,
			cartridgePath,
			this.settings.encoder || undefined,
			this.settings.checkpoint || undefined,
			(event) => {
				modal.update(event);
				this.onProgress(event);
			},
		);

		if (!result.success) {
			modal.close();
			this.sidebarView?.hideProgress();
			new Notice(`Resonance Lattice: Sync failed.\n${result.output}`, 10000);
		}

		// Restart server with updated cartridge
		if (wasRunning || result.success) {
			await this.startServer();
		}
	}

	private async rebuildAndRestart(): Promise<void> {
		this.processManager.stopServer();
		this.updateStatusBar("offline");
		await this.buildCartridge();
		await this.startServer();
	}

	// ── Cancel ──────────────────────────────────────────────────────

	cancelBuild(): void {
		if (this.processManager.isBuildRunning) {
			this.processManager.cancelBuild();
			this.sidebarView?.hideProgress();
			new Notice("Resonance Lattice: Build cancelled.");
		} else {
			new Notice("No build in progress.");
		}
	}

	// ── Sidebar ─────────────────────────────────────────────────────

	private async activateSidebar(): Promise<void> {
		const existing = this.app.workspace.getLeavesOfType(VIEW_TYPE);
		if (existing.length > 0) {
			this.app.workspace.revealLeaf(existing[0]);
			return;
		}
		const leaf = this.app.workspace.getRightLeaf(false);
		if (leaf) {
			await leaf.setViewState({ type: VIEW_TYPE, active: true });
			this.app.workspace.revealLeaf(leaf);
		}
	}

	// ── Live file updates via HTTP endpoints ────────────────────────

	private onFileChanged(file: TFile): void {
		if (!this.settings.autoSync) return;
		this.pendingChanges.add(file.path);
		this.scheduleLiveSync();
	}

	private onFileDeleted(file: TFile): void {
		if (!this.settings.autoSync) return;
		this.pendingDeletes.add(file.path);
		this.pendingChanges.delete(file.path);
		this.scheduleLiveSync();
	}

	private scheduleLiveSync(): void {
		if (this.syncTimer) {
			clearTimeout(this.syncTimer);
		}
		// Debounce: wait 5 seconds after last change before syncing
		this.syncTimer = setTimeout(() => this.executeLiveSync(), 5000);
	}

	private async executeLiveSync(): Promise<void> {
		const alive = await this.client.health();
		if (!alive) return;

		// Process deletions: remove old source IDs from the running server
		let deletedCount = 0;
		for (const filePath of this.pendingDeletes) {
			try {
				const result = await this.client.remove(filePath);
				if (result?.removed) deletedCount++;
			} catch (err) {
				console.error(`Resonance Lattice: failed to remove ${filePath}:`, err);
			}
		}

		// Process additions/updates: remove stale chunks then POST /add
		let addedCount = 0;
		for (const filePath of this.pendingChanges) {
			const file = this.app.vault.getAbstractFileByPath(filePath);
			if (!(file instanceof TFile)) continue;

			try {
				// Remove old source entry first to avoid duplicate/stale chunks
				await this.client.remove(filePath);
				const content = await this.app.vault.cachedRead(file);
				const result = await this.client.add(
					content,
					filePath,
					1.0,
					{ source_file: filePath, heading: "" },
				);
				if (result) addedCount++;
			} catch (err) {
				console.error(`Resonance Lattice: failed to live-add ${filePath}:`, err);
			}
		}

		if (addedCount > 0 || deletedCount > 0) {
			this.updateStatusBarFromServer();
		}

		this.pendingChanges.clear();
		this.pendingDeletes.clear();
	}

	// ── Server lifecycle ────────────────────────────────────────────

	private async startServer(): Promise<boolean> {
		const cartridgePath = this.getCartridgePath();
		if (!existsSync(cartridgePath)) {
			this.updateStatusBar("no cartridge");
			return false;
		}

		new Notice("Resonance Lattice: Starting server\u2026");

		const ok = await this.processManager.startServer(
			this.settings.rlatPath,
			cartridgePath,
			this.settings.port,
			this.settings.encoder || undefined,
			this.settings.checkpoint || undefined,
		);

		if (ok) {
			await this.updateStatusBarFromServer();
			return true;
		} else {
			this.updateStatusBar("offline");
			new Notice(
				"Resonance Lattice: Failed to start server. Check that rlat is installed (pip install resonance-lattice).",
				10000,
			);
			return false;
		}
	}

	private async tryAutoStart(): Promise<void> {
		const cartridgePath = this.getCartridgePath();
		if (!existsSync(cartridgePath)) {
			this.updateStatusBar("no cartridge");
			return;
		}

		// Check if server is already running
		const alive = await this.client.health();
		if (alive) {
			await this.updateStatusBarFromServer();
			return;
		}

		await this.startServer();
	}

	// ── Info ─────────────────────────────────────────────────────────

	private async showServerInfo(): Promise<void> {
		const info = await this.client.info();
		if (!info) {
			new Notice("Resonance Lattice: Server is not running.");
			return;
		}
		new Notice(
			`Resonance Lattice\n` +
				`Sources: ${info.source_count}\n` +
				`Field: ${info.field_type} (${info.bands}B \u00d7 ${info.dim}D)\n` +
				`Size: ${info.field_size_mb.toFixed(1)} MB` +
				(info.snr ? `\nSNR: ${info.snr.toFixed(1)}` : ""),
			8000,
		);
	}

	// ── Status bar ──────────────────────────────────────────────────

	private updateStatusBar(status: string): void {
		if (this.statusBarEl) {
			this.statusBarEl.setText(`RL: ${status}`);
		}
	}

	private async updateStatusBarFromServer(): Promise<void> {
		const info = await this.client.info();
		const count = info?.source_count ?? "?";
		this.updateStatusBar(`${count} sources`);
		if (info) {
			new Notice(`Resonance Lattice: Server running (${count} sources)`);
		}
	}
}
