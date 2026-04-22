import { App, PluginSettingTab, Setting } from "obsidian";
import type ResonanceLatticePlugin from "./main";

export interface ResonanceLatticeSettings {
	rlatPath: string;
	cartridgeMode: "vault" | "external";
	externalCartridgePath: string;
	port: number;
	topK: number;
	enableCascade: boolean;
	cascadeDepth: number;
	enableContradictions: boolean;
	debounceMs: number;
	autoSync: boolean;
	encoder: string;
	checkpoint: string;
	defaultGraphView: boolean;
	similarityThreshold: number;
}

export const DEFAULT_SETTINGS: ResonanceLatticeSettings = {
	rlatPath: "rlat",
	cartridgeMode: "vault",
	externalCartridgePath: "",
	port: 27182,
	topK: 10,
	enableCascade: true,
	cascadeDepth: 2,
	enableContradictions: false,
	debounceMs: 300,
	autoSync: false,
	encoder: "intfloat/e5-large-v2",
	checkpoint: "",
	defaultGraphView: false,
	similarityThreshold: 0.7,
};

export class ResonanceLatticeSettingTab extends PluginSettingTab {
	constructor(app: App, private plugin: ResonanceLatticePlugin) {
		super(app, plugin);
	}

	display(): void {
		const { containerEl } = this;
		containerEl.empty();

		containerEl.createEl("h2", { text: "Resonance Lattice" });

		new Setting(containerEl)
			.setName("rlat binary path")
			.setDesc("Path to the rlat CLI binary. Use 'rlat' if installed globally via pip.")
			.addText((text) =>
				text
					.setPlaceholder("rlat")
					.setValue(this.plugin.settings.rlatPath)
					.onChange(async (value) => {
						this.plugin.settings.rlatPath = value.trim() || "rlat";
						await this.plugin.saveSettings();
					}),
			);

		new Setting(containerEl)
			.setName("Cartridge source")
			.setDesc("Build a cartridge from this vault, or use an existing .rlat file.")
			.addDropdown((dropdown) =>
				dropdown
					.addOption("vault", "Build from vault")
					.addOption("external", "Use existing .rlat file")
					.setValue(this.plugin.settings.cartridgeMode)
					.onChange(async (value) => {
						this.plugin.settings.cartridgeMode = value as "vault" | "external";
						await this.plugin.saveSettings();
						this.display();
					}),
			);

		if (this.plugin.settings.cartridgeMode === "external") {
			new Setting(containerEl)
				.setName("Cartridge file path")
				.setDesc("Absolute path to your .rlat cartridge file.")
				.addText((text) =>
					text
						.setPlaceholder("/path/to/my-data.rlat")
						.setValue(this.plugin.settings.externalCartridgePath)
						.onChange(async (value) => {
							this.plugin.settings.externalCartridgePath = value.trim();
							await this.plugin.saveSettings();
						}),
				);
		}

		new Setting(containerEl)
			.setName("Server port")
			.setDesc("Port for the local rlat serve process.")
			.addText((text) =>
				text
					.setPlaceholder("27182")
					.setValue(String(this.plugin.settings.port))
					.onChange(async (value) => {
						const port = parseInt(value, 10);
						if (!isNaN(port) && port > 0 && port < 65536) {
							this.plugin.settings.port = port;
							await this.plugin.saveSettings();
						}
					}),
			);

		new Setting(containerEl)
			.setName("Results count")
			.setDesc("Number of results to return per search.")
			.addText((text) =>
				text
					.setPlaceholder("10")
					.setValue(String(this.plugin.settings.topK))
					.onChange(async (value) => {
						const k = parseInt(value, 10);
						if (!isNaN(k) && k > 0 && k <= 50) {
							this.plugin.settings.topK = k;
							await this.plugin.saveSettings();
						}
					}),
			);

		new Setting(containerEl)
			.setName("Show related topics")
			.setDesc("Enable multi-hop cascade to discover related topics beyond direct results.")
			.addToggle((toggle) =>
				toggle.setValue(this.plugin.settings.enableCascade).onChange(async (value) => {
					this.plugin.settings.enableCascade = value;
					await this.plugin.saveSettings();
				}),
			);

		new Setting(containerEl)
			.setName("Cascade depth")
			.setDesc("How many hops to follow for related topics (1\u20133).")
			.addText((text) =>
				text
					.setPlaceholder("2")
					.setValue(String(this.plugin.settings.cascadeDepth))
					.onChange(async (value) => {
						const d = parseInt(value, 10);
						if (!isNaN(d) && d >= 1 && d <= 3) {
							this.plugin.settings.cascadeDepth = d;
							await this.plugin.saveSettings();
						}
					}),
			);

		new Setting(containerEl)
			.setName("Detect contradictions")
			.setDesc("Surface conflicting information between sources. Adds ~5-10ms per search.")
			.addToggle((toggle) =>
				toggle.setValue(this.plugin.settings.enableContradictions).onChange(async (value) => {
					this.plugin.settings.enableContradictions = value;
					await this.plugin.saveSettings();
				}),
			);

		new Setting(containerEl)
			.setName("Search debounce (ms)")
			.setDesc("Delay before sending search request after typing stops.")
			.addText((text) =>
				text
					.setPlaceholder("300")
					.setValue(String(this.plugin.settings.debounceMs))
					.onChange(async (value) => {
						const ms = parseInt(value, 10);
						if (!isNaN(ms) && ms >= 50 && ms <= 2000) {
							this.plugin.settings.debounceMs = ms;
							await this.plugin.saveSettings();
						}
					}),
			);

		new Setting(containerEl)
			.setName("Default to graph view")
			.setDesc("Open search results in graph view instead of list view.")
			.addToggle((toggle) =>
				toggle.setValue(this.plugin.settings.defaultGraphView).onChange(async (value) => {
					this.plugin.settings.defaultGraphView = value;
					await this.plugin.saveSettings();
				}),
			);

		new Setting(containerEl)
			.setName("Graph similarity threshold")
			.setDesc("Minimum band-score cosine similarity to draw edges between results (0.0\u20131.0).")
			.addText((text) =>
				text
					.setPlaceholder("0.7")
					.setValue(String(this.plugin.settings.similarityThreshold))
					.onChange(async (value) => {
						const t = parseFloat(value);
						if (!isNaN(t) && t >= 0 && t <= 1) {
							this.plugin.settings.similarityThreshold = t;
							await this.plugin.saveSettings();
						}
					}),
			);

		new Setting(containerEl)
			.setName("Encoder")
			.setDesc("Sentence-transformer model for encoding. Default: intfloat/e5-large-v2.")
			.addText((text) =>
				text
					.setPlaceholder("intfloat/e5-large-v2")
					.setValue(this.plugin.settings.encoder)
					.onChange(async (value) => {
						this.plugin.settings.encoder = value.trim();
						await this.plugin.saveSettings();
					}),
			);

		new Setting(containerEl)
			.setName("Encoder checkpoint")
			.setDesc("Optional path to trained semantic heads checkpoint.")
			.addText((text) =>
				text
					.setPlaceholder("(none)")
					.setValue(this.plugin.settings.checkpoint)
					.onChange(async (value) => {
						this.plugin.settings.checkpoint = value.trim();
						await this.plugin.saveSettings();
					}),
			);

		if (this.plugin.settings.cartridgeMode === "vault") {
			new Setting(containerEl)
				.setName("Auto-sync on file changes")
				.setDesc(
					"Incrementally sync the cartridge when vault files change (add new, update changed, remove deleted). Uses rlat sync.",
				)
				.addToggle((toggle) =>
					toggle.setValue(this.plugin.settings.autoSync).onChange(async (value) => {
						this.plugin.settings.autoSync = value;
						await this.plugin.saveSettings();
					}),
				);
		}
	}
}
