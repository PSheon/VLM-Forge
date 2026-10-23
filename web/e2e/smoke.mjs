// UI smoke test against a running `forge serve` (docs/ui-spec §1). Read-only: it navigates,
// opens and closes popovers, and presses Esc — it never saves an answer or runs inference.
//
//   cd web && npm run e2e                       # http://127.0.0.1:8796
//   FORGE_URL=http://host:8796 npm run e2e
//   npm run e2e -- --shots ../e2e-shots          # also save one screenshot per view
//
// Per view, in light 1440×900, dark 1440×900 and narrow 1024×768:
//   - exactly one visible orange (primary) button, and it lives in the action column
//   - no tab bars
//   - wide layouts: the title row and the action column's tool row start level and are as
//     tall; narrow: the primary is pinned inside the viewport
//   - no console errors or page errors
// Sub-views (review, diff, playground) must go back on Esc; the system popover must close on
// an outside click. Views that need data (a dataset, pending questions, a finished run) are
// skipped with a note when the workspace does not have it.

import { chromium } from "playwright";

const BASE = process.env.FORGE_URL ?? "http://127.0.0.1:8796";
const shotsAt = process.argv.indexOf("--shots");
const SHOTS = shotsAt > 0 ? process.argv[shotsAt + 1] : null;

const CONFIGS = [
  { tag: "light", viewport: { width: 1440, height: 900 }, colorScheme: "light", wide: true },
  { tag: "dark", viewport: { width: 1440, height: 900 }, colorScheme: "dark", wide: true },
  { tag: "narrow", viewport: { width: 1024, height: 768 }, colorScheme: "light", wide: false },
];

const failures = [];
const notes = [];
const fail = (where, what) => failures.push(`${where}: ${what}`);

async function checkView(page, cfg, name, errors) {
  const where = `${cfg.tag}/${name}`;
  await page.waitForTimeout(300);
  const primaries = page.locator("button.primary:visible, a.btn.primary:visible");
  const n = await primaries.count();
  if (n !== 1) fail(where, `${n} visible primary buttons (want 1)`);
  if (n >= 1 && !(await primaries.first().evaluate((el) => !!el.closest(".actions")))) fail(where, "primary button is not in the action column");
  const tabs = await page.locator(".tabs.big:visible").count();
  if (tabs) fail(where, `${tabs} tab bar(s)`);
  if (cfg.wide) {
    const head = await page.locator(".page-head").first().boundingBox();
    const tool = await page.locator(".actions-head").first().boundingBox();
    if (!head || !tool) fail(where, "missing title row or tool row");
    else if (Math.abs(head.y - tool.y) > 1 || Math.abs(head.height - tool.height) > 1) fail(where, `title row ${head.y}+${head.height} vs tool row ${tool.y}+${tool.height}`);
  } else if (n >= 1) {
    const b = await primaries.first().boundingBox();
    if (!b || b.y + b.height > cfg.viewport.height || b.y < 0) fail(where, "primary button not inside the viewport");
  }
  if (errors.length) fail(where, `console: ${errors.splice(0).join(" | ").slice(0, 300)}`);
  if (SHOTS) await page.screenshot({ path: `${SHOTS}/${cfg.tag}-${name}.png` });
}

async function run(browser, cfg) {
  const ctx = await browser.newContext({ viewport: cfg.viewport, colorScheme: cfg.colorScheme });
  await ctx.addInitScript(() => { try { localStorage.setItem("forge.name", "e2e"); } catch { /* private mode */ } });
  const page = await ctx.newPage();
  const errors = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  page.on("console", (m) => { if (m.type() === "error") errors.push(m.text()); });
  const step = (k) => page.click(`nav.steps a[href='#${k}']`).then(() => page.waitForTimeout(1500));

  await page.goto(`${BASE}/#datasets`, { waitUntil: "networkidle", timeout: 60000 });
  await page.waitForTimeout(1000);
  await checkView(page, cfg, "datasets", errors);

  // system popover: opens, closes on an outside click
  await page.locator(".sys .pill").click();
  if (!(await page.locator(".sys-pop").count())) fail(`${cfg.tag}/system`, "popover did not open");
  await page.mouse.click(5, cfg.viewport.height - 5);
  await page.waitForTimeout(200);
  if (await page.locator(".sys-pop").count()) fail(`${cfg.tag}/system`, "popover did not close on an outside click");

  if (await page.locator(".ds-row").count()) {
    await page.locator(".ds-row").first().click();
    await page.waitForTimeout(2000);
    await checkView(page, cfg, "dataset", errors);
    const cta = page.locator(".actions .primary.big").first();
    if ((await cta.innerText()).startsWith("去檢查")) {
      await cta.click();
      await page.waitForTimeout(3000);
      await checkView(page, cfg, "review", errors);
      await page.keyboard.press("Escape");
      await page.waitForTimeout(800);
      if (!(await page.locator(".drop").count())) fail(`${cfg.tag}/review`, "Esc did not go back to the image grid");
    } else notes.push(`${cfg.tag}: no pending questions, review skipped`);
  } else notes.push(`${cfg.tag}: no datasets, dataset and review skipped`);

  await step("train");
  await checkView(page, cfg, "train", errors);

  await step("test");
  await checkView(page, cfg, "scores", errors);
  const toDiff = page.locator(".actions .primary.big").first();
  if ((await toDiff.innerText()).startsWith("看")) {
    await toDiff.click();
    await page.waitForTimeout(2500);
    await checkView(page, cfg, "diff", errors);
    await page.keyboard.press("Escape");
    await page.waitForTimeout(800);
    if (!(await page.locator(".suite").count())) fail(`${cfg.tag}/diff`, "Esc did not go back to the scores");
  } else notes.push(`${cfg.tag}: no finished run, diff skipped`);
  const toPlay = page.locator(".actions button.ghost", { hasText: "試一張圖" });
  if (await toPlay.count()) {
    await toPlay.click();
    await page.waitForTimeout(2000);
    await checkView(page, cfg, "play", errors);
    await page.keyboard.press("Escape");
    await page.waitForTimeout(800);
  }
  await ctx.close();
}

const browser = await chromium.launch({ args: ["--disable-dev-shm-usage"] });
try {
  for (const cfg of CONFIGS) await run(browser, cfg);
} finally {
  await browser.close();
}
for (const n of notes) console.log(`note  ${n}`);
if (failures.length) {
  for (const f of failures) console.log(`FAIL  ${f}`);
  console.log(`\n${failures.length} failure(s)`);
  process.exit(1);
}
console.log("ok    UI smoke test passed");
