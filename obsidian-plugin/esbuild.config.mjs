import esbuild from "esbuild";
import process from "process";

const watch = process.argv.includes("--watch");

const context = await esbuild.context({
	entryPoints: ["src/main.ts"],
	bundle: true,
	external: ["obsidian", "electron", "@codemirror/*", "@lezer/*"],
	format: "cjs",
	target: "es2018",
	platform: "node",
	outfile: "main.js",
	sourcemap: "inline",
	treeShaking: true,
	logLevel: "info",
});

if (watch) {
	await context.watch();
} else {
	await context.rebuild();
	process.exit(0);
}
