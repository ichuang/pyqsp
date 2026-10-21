const assert = require('node:assert/strict');
const test = require('node:test');
const requestReview = require('./request-review.cjs');

function fixture() {
  const run = {
    id: 10, workflow_id: 5, run_attempt: 1, event: 'pull_request', conclusion: 'success',
    status: 'completed', path: '.github/workflows/run_tests.yml',
    head_sha: 'tested-head', head_branch: 'fix/example',
    pull_requests: [{number: 42, base: {ref: 'master', sha: 'tested-base'}}],
  };
  const pull = {
    number: 42, state: 'open', draft: false, user: {login: 'contributor'},
    head: {sha: 'tested-head', ref: 'fix/example'},
    base: {ref: 'master', sha: 'tested-base', repo: {full_name: 'ichuang/pyqsp'}},
    requested_reviewers: [],
  };
  const state = {run, pull, currentRun: {...run}, recentRuns: [run], reviews: [], requests: []};
  const github = {
    rest: {
      actions: {
        getWorkflowRun: async () => ({data: state.currentRun}),
        listWorkflowRuns: async () => ({data: {workflow_runs: state.recentRuns}}),
      },
      pulls: {
        list: 'pulls.list', listReviews: 'pulls.listReviews',
        get: async () => ({data: state.pull}),
        requestReviewers: async args => state.requests.push(args),
      },
    },
    paginate: async method => {
      if (method === 'pulls.list') return [state.pull];
      if (method === 'pulls.listReviews') return state.reviews;
      throw new Error(`Unexpected API call: ${method}`);
    },
  };
  state.invoke = () => requestReview({
    github, reviewer: 'hironori-kondo', core: {info() {}},
    context: {repo: {owner: 'ichuang', repo: 'pyqsp'}, payload: {workflow_run: run}},
  });
  return state;
}

test('requests the maintainer after the current commit passes', async () => {
  const state = fixture();
  await state.invoke();
  assert.deepEqual(state.requests, [{
    owner: 'ichuang', repo: 'pyqsp', pull_number: 42, reviewers: ['hironori-kondo'],
  }]);
});

test('resolves fork PRs even when workflow_run has no pull_requests', async () => {
  const state = fixture();
  state.run.pull_requests = [];
  await state.invoke();
  assert.equal(state.requests.length, 1);
});

for (const [name, change] of [
  ['failed tests', s => { s.run.conclusion = 'failure'; }],
  ['cancelled tests', s => { s.run.conclusion = 'cancelled'; }],
  ['push runs', s => { s.run.event = 'push'; }],
  ['unrelated workflow', s => { s.run.path = '.github/workflows/other.yml'; }],
  ['drafts', s => { s.pull.draft = true; }],
  ['closed PRs', s => { s.pull.state = 'closed'; }],
  ['a newer PR commit', s => { s.pull.head.sha = 'untested-head'; }],
  ['a different PR with the same commit', s => { s.run.pull_requests[0].number = 43; }],
  ['a retargeted PR', s => { s.pull.base.ref = 'other-base'; }],
  ['a changed base commit', s => { s.pull.base.sha = 'new-base'; }],
  ['a different source branch', s => { s.pull.head.ref = 'other-branch'; }],
  ['a different repository', s => { s.pull.base.repo.full_name = 'other/pyqsp'; }],
  ['self review', s => { s.pull.user.login = 'Hironori-Kondo'; }],
  ['a newer run attempt', s => { s.currentRun.run_attempt = 2; }],
  ['a rerun still in progress', s => { s.currentRun.status = 'in_progress'; }],
  ['a failed rerun', s => { s.currentRun.conclusion = 'failure'; }],
  ['a newer run for the same commit', s => {
    s.recentRuns.push({...s.run, id: 11, status: 'in_progress', conclusion: null});
  }],
  ['an existing request', s => {
    s.pull.requested_reviewers = [{login: 'Hironori-Kondo'}];
  }],
  ['an existing review of this commit', s => {
    s.reviews = [{user: {login: 'hironori-kondo'}, commit_id: 'tested-head', state: 'APPROVED'}];
  }],
]) {
  test(`does not request review for ${name}`, async () => {
    const state = fixture();
    change(state);
    await state.invoke();
    assert.deepEqual(state.requests, []);
  });
}

test('requests another review after the reviewed commit changes', async () => {
  const state = fixture();
  state.reviews = [{user: {login: 'hironori-kondo'}, commit_id: 'old-head', state: 'APPROVED'}];
  await state.invoke();
  assert.equal(state.requests.length, 1);
});
