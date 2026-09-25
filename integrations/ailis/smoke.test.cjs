const assert = require('node:assert/strict');
const { AILISEmberHarness } = require('./ailis-ember-harness.cjs');

(async () => {
  const harness = new AILISEmberHarness({ mode: 'enforce' });
  const first = await harness.check({
    runId: 'public-smoke', stage: 'input_parse', text: 'neutral task',
    evaluator: async () => ({ decision: 'allow', riskLevel: 'none' }),
  });
  assert.equal(first.blocked, false);
  const second = await harness.check({
    runId: 'public-smoke', stage: 'retrieval', text: 'risk test fixture',
    evaluator: async () => ({ decision: 'block', riskLevel: 'high' }),
  });
  assert.equal(second.blocked, true);
  assert.ok(second.rollbackTo);
  assert.equal(second.rollbackTo.stage, 'input_parse');
  assert.equal(harness.listRunRecords('public-smoke').length, 2);
  console.log('PASS: actual AILIS module allows a clean stage and rolls back a blocked stage.');
})().catch(error => { console.error(error); process.exitCode = 1; });
