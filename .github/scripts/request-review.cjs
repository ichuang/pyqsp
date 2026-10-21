// Runs only from the default branch, with no code or artifacts from the PR.
module.exports = async function requestReview({github, context, core, reviewer}) {
  const run = context.payload.workflow_run;
  if (run.event !== 'pull_request' || run.conclusion !== 'success' ||
      run.path !== '.github/workflows/run_tests.yml') return;

  const repo = context.repo;
  const {data: currentRun} = await github.rest.actions.getWorkflowRun({
    ...repo, run_id: run.id,
  });
  // A delayed event must not request review during a newer, failing rerun.
  if (currentRun.run_attempt !== run.run_attempt ||
      currentRun.status !== 'completed' || currentRun.conclusion !== 'success') return;

  const {data: recent} = await github.rest.actions.listWorkflowRuns({
    ...repo, workflow_id: run.workflow_id, event: 'pull_request', head_sha: run.head_sha,
    per_page: 100,
  });
  // Retargeting or marking ready can start a new run without changing the SHA.
  if (recent.workflow_runs.some(other => other.id > run.id &&
      other.head_branch === run.head_branch)) return;

  // workflow_run.pull_requests can be empty for fork PRs. Resolve candidates
  // through GitHub instead of trusting a PR number supplied by an artifact.
  const pulls = await github.paginate(github.rest.pulls.list, {...repo, state: 'open'});
  for (const candidate of pulls) {
    if (candidate.head.sha !== run.head_sha) continue;
    if (run.pull_requests.length &&
        !run.pull_requests.some(pr => pr.number === candidate.number)) continue;
    const args = {...repo, pull_number: candidate.number};
    const {data: pull} = await github.rest.pulls.get(args);
    if (pull.state !== 'open' || pull.draft || pull.head.sha !== run.head_sha ||
        pull.base.repo.full_name !== `${repo.owner}/${repo.repo}` ||
        pull.head.ref !== run.head_branch ||
        pull.user.login.toLowerCase() === reviewer.toLowerCase()) continue;

    const testedPull = (run.pull_requests || []).find(pr => pr.number === pull.number);
    if (testedPull && (testedPull.base.ref !== pull.base.ref ||
                       testedPull.base.sha !== pull.base.sha)) continue;
    if (pull.requested_reviewers.some(user =>
      user.login.toLowerCase() === reviewer.toLowerCase())) continue;

    const reviews = await github.paginate(github.rest.pulls.listReviews, args);
    if (reviews.some(review => review.user &&
        review.user.login.toLowerCase() === reviewer.toLowerCase() &&
        review.commit_id === pull.head.sha && review.state !== 'DISMISSED')) continue;

    await github.rest.pulls.requestReviewers({...args, reviewers: [reviewer]});
    core.info(`Requested ${reviewer}'s review of PR #${pull.number} after Run Tests passed.`);
  }
};
