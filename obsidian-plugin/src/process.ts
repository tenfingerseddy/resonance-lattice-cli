import { spawn, ChildProcess } from "child_process";
import { Notice } from "obsidian";
import type { LatticeClient } from "./client";
import type { ProgressEvent } from "./progress-modal";

export class ProcessManager {
	private serverProcess: ChildProcess | null = null;
	private buildProcess: ChildProcess | null = null;

	constructor(private client: LatticeClient) {}

	get isServerRunning(): boolean {
		return this.serverProcess !== null && this.serverProcess.exitCode === null;
	}

	get isBuildRunning(): boolean {
		return this.buildProcess !== null && this.buildProcess.exitCode === null;
	}

	async startServer(
		rlatPath: string,
		cartridgePath: string,
		port: number,
		encoder?: string,
		checkpoint?: string,
	): Promise<boolean> {
		if (this.isServerRunning) {
			return true;
		}

		const args = ["serve", cartridgePath, "--port", String(port), "--host", "127.0.0.1"];
		if (encoder) args.push("--encoder", encoder);
		if (checkpoint) args.push("--checkpoint", checkpoint);

		return new Promise((resolve) => {
			try {
				this.serverProcess = spawn(rlatPath, args, {
					stdio: ["ignore", "pipe", "pipe"],
					detached: false,
				});
			} catch (err) {
				console.error("Resonance Lattice: failed to spawn server:", err);
				resolve(false);
				return;
			}

			this.serverProcess.stdout?.on("data", (data: Buffer) => {
				console.log("rlat serve:", data.toString().trim());
			});

			this.serverProcess.stderr?.on("data", (data: Buffer) => {
				console.error("rlat serve:", data.toString().trim());
			});

			this.serverProcess.on("error", (err) => {
				console.error("Resonance Lattice: server process error:", err);
				this.serverProcess = null;
				resolve(false);
			});

			this.serverProcess.on("exit", (code) => {
				console.log(`Resonance Lattice: server exited with code ${code}`);
				this.serverProcess = null;
			});

			this.waitForHealth(resolve);
		});
	}

	private async waitForHealth(resolve: (ok: boolean) => void): Promise<void> {
		const delays = [200, 400, 800, 1600, 3200];
		for (const delay of delays) {
			await sleep(delay);
			if (!this.isServerRunning) {
				resolve(false);
				return;
			}
			const ok = await this.client.health();
			if (ok) {
				resolve(true);
				return;
			}
		}
		resolve(false);
	}

	stopServer(): void {
		if (this.serverProcess && this.serverProcess.exitCode === null) {
			this.serverProcess.kill("SIGTERM");
			setTimeout(() => {
				if (this.serverProcess && this.serverProcess.exitCode === null) {
					this.serverProcess.kill("SIGKILL");
				}
			}, 3000);
		}
		this.serverProcess = null;
	}

	async build(
		rlatPath: string,
		vaultPath: string,
		cartridgePath: string,
		encoder?: string,
		checkpoint?: string,
		onProgress?: (event: ProgressEvent) => void,
	): Promise<{ success: boolean; output: string }> {
		return this.runCli(
			rlatPath,
			["build", vaultPath, "-o", cartridgePath, "--progress"],
			encoder,
			checkpoint,
			onProgress,
		);
	}

	async sync(
		rlatPath: string,
		vaultPath: string,
		cartridgePath: string,
		encoder?: string,
		checkpoint?: string,
		onProgress?: (event: ProgressEvent) => void,
	): Promise<{ success: boolean; output: string }> {
		return this.runCli(
			rlatPath,
			["sync", cartridgePath, vaultPath, "--progress"],
			encoder,
			checkpoint,
			onProgress,
		);
	}

	private async runCli(
		rlatPath: string,
		baseArgs: string[],
		encoder?: string,
		checkpoint?: string,
		onProgress?: (event: ProgressEvent) => void,
	): Promise<{ success: boolean; output: string }> {
		if (this.isBuildRunning) {
			return { success: false, output: "A build is already in progress." };
		}

		const args = [...baseArgs];
		if (encoder) args.push("--encoder", encoder);
		if (checkpoint) args.push("--checkpoint", checkpoint);

		return new Promise((resolve) => {
			let output = "";
			let errOutput = "";
			let stdoutBuffer = "";

			try {
				this.buildProcess = spawn(rlatPath, args, {
					stdio: ["ignore", "pipe", "pipe"],
					detached: false,
				});
			} catch (err) {
				resolve({
					success: false,
					output: `Failed to start: ${err}`,
				});
				return;
			}

			this.buildProcess.stdout?.on("data", (data: Buffer) => {
				stdoutBuffer += data.toString();
				// Process complete JSON lines
				const lines = stdoutBuffer.split("\n");
				stdoutBuffer = lines.pop() ?? "";
				for (const line of lines) {
					const trimmed = line.trim();
					if (!trimmed) continue;
					output += trimmed + "\n";
					if (onProgress) {
						try {
							const event: ProgressEvent = JSON.parse(trimmed);
							onProgress(event);
						} catch {
							// Not JSON — show as notice
							new Notice(`RL: ${trimmed}`, 3000);
						}
					} else {
						new Notice(`RL: ${trimmed}`, 3000);
					}
				}
			});

			this.buildProcess.stderr?.on("data", (data: Buffer) => {
				errOutput += data.toString();
			});

			this.buildProcess.on("error", (err) => {
				this.buildProcess = null;
				resolve({ success: false, output: `Error: ${err.message}` });
			});

			this.buildProcess.on("exit", (code) => {
				this.buildProcess = null;
				if (code === 0) {
					resolve({ success: true, output: output.trim() });
				} else {
					resolve({
						success: false,
						output: errOutput.trim() || `Failed with exit code ${code}`,
					});
				}
			});
		});
	}

	cancelBuild(): void {
		if (this.buildProcess && this.buildProcess.exitCode === null) {
			this.buildProcess.kill("SIGTERM");
			setTimeout(() => {
				if (this.buildProcess && this.buildProcess.exitCode === null) {
					this.buildProcess.kill("SIGKILL");
				}
			}, 2000);
		}
		this.buildProcess = null;
	}

	destroy(): void {
		this.stopServer();
		this.cancelBuild();
	}
}

function sleep(ms: number): Promise<void> {
	return new Promise((resolve) => setTimeout(resolve, ms));
}
