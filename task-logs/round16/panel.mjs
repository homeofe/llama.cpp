import { readFileSync, writeFileSync, appendFileSync } from 'node:fs';
import { createHash } from 'node:crypto';

const out = process.argv[2];
const source = readFileSync('/opt/llm-gateway/apps/bench/dist/hard.js', 'utf8');
const checks = await import('data:text/javascript;base64,' + Buffer.from(source + '\nexport { CODE, CHECKS, TOOLS, codeOf, toolLoop };').toString('base64'));
writeFileSync(`${out}/panel-source.json`, JSON.stringify({ path: '/opt/llm-gateway/apps/bench/dist/hard.js', sha256: createHash('sha256').update(source).digest('hex'), excluded: ['long-combine: exceeds the measured 8192 context'], seed: 1, thinking: false }, null, 2));
async function chat(body) {
  const response = await fetch('http://127.0.0.1:18796/v1/chat/completions', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ...body, stream: false, temperature: 0.6, top_p: 0.95, seed: 1, cache_prompt: false, chat_template_kwargs: { enable_thinking: false } }),
  });
  if (!response.ok) throw new Error(`HTTP ${response.status}: ${await response.text()}`);
  return response.json();
}
let total = 0, passed = 0;
async function run(id, category, check) {
  const start = Date.now();
  let right = false, detail;
  try { [right, detail] = await check(); } catch (error) { detail = String(error); }
  total++; passed += Number(right);
  const row = { id, category, right, detail, seconds: (Date.now()-start)/1000 };
  appendFileSync(`${out}/panel-results.jsonl`, JSON.stringify(row)+'\n');
  console.log(JSON.stringify(row));
}
for (const test of checks.CODE) {
  await run(test.id, 'code', async () => {
    const response = await chat({ messages: [{ role: 'user', content: test.prompt + '\nReturn only the Python code.' }], max_tokens: 8192 });
    const answer = checks.strip(response.choices[0]?.message?.content ?? '');
    const result = await checks.runPython(checks.codeOf(answer) + '\n' + test.tests + '\nprint("ALL TESTS PASSED")');
    return [result.ok, result.output];
  });
}
for (const test of checks.TOOLS) {
  await run(test.id, 'tools', async () => {
    const { final, calls } = await checks.toolLoop(chat, 'round16', test);
    return [test.check(final, calls), { final, calls }];
  });
}
for (const test of checks.CHECKS.filter(test => test.category !== 'long')) {
  await run(test.id, test.category, async () => {
    const response = await chat({ ...test.body(''), max_tokens: 8192 });
    const answer = response.choices[0]?.message?.content ?? '';
    return [test.check(answer), answer];
  });
}
writeFileSync(`${out}/panel-verdict.json`, JSON.stringify({ passed, total }, null, 2));
console.log(JSON.stringify({ complete: true, passed, total }));
