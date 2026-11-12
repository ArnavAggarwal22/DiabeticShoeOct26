// Cross-platform runner for the Python ML service: `node scripts/ml.mjs <setup|dev|test>`.
import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import path from "node:path";

const root = path.resolve(path.dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, "$1")), "..");
const mlDir = path.join(root, "ml");
const win = process.platform === "win32";
const venvPy = path.join(mlDir, ".venv", win ? "Scripts" : "bin", win ? "python.exe" : "python");
const run = (cmd, args, cwd = mlDir) => {
  const r = spawnSync(cmd, args, { cwd, stdio: "inherit" });
  if (r.status !== 0) process.exit(r.status ?? 1);
};

const task = process.argv[2] || "dev";
if (task === "setup") {
  if (!existsSync(venvPy)) run(win ? "python" : "python3", ["-m", "venv", ".venv"]);
  run(venvPy, ["-m", "pip", "install", "-r", "requirements-dev.txt"]);
} else {
  if (!existsSync(venvPy)) {
    console.error("ML virtualenv not found. Run `npm run ml:setup` first.");
    process.exit(1);
  }
  if (task === "test") run(venvPy, ["-m", "pytest", "-q"]);
  else run(venvPy, ["-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", process.env.ML_PORT || "8000", ...(task === "dev" ? ["--reload"] : [])]);
}
