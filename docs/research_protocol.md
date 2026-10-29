# DSA5101 HetRec Last.fm research protocol

**Status: APPROVED — lastfm-v1.** This approved design is not proof of successful implementation, numerical execution, or library compatibility. Implementation and test evaluation remain subject to the validation gates below.

**Protocol identifier:** `lastfm-v1`

## Scope, question, and evidential limits

**Research question.** Do friendship-graph communities improve held-out artist recommendation beyond listening-history-based matrix factorization, particularly for users with sparse *observed training* histories?

The supplied HetRec 2011 Last.fm archive contains user--artist aggregate listening counts and a reciprocal friendship graph. It has no listening-event timestamps. Consequently, all train/validation/test partitions are random-within-user partitions of user--artist pairs; they are not forecasts, causal interventions, or temporal tests. Friendship, community membership, and listening similarity may all reflect unmeasured common causes. Results can establish an association in this dataset only, not that friendship communities cause better recommendations, social influence, or generalizable cold-start performance.

### Confirmed source facts

- The source is `data/raw/hetrec2011-lastfm-2k.zip`; its expected members, ZIP CRCs, schemas, keys, and audited counts are validated before analysis.
- `user_artists.dat` has 92,834 unique positive-count user--artist pairs for 1,892 users and 17,632 artists. It has no listening timestamps.
- `user_friends.dat` has reciprocal directed records and canonically represents 12,717 undirected friendships.
- Tags and tag timestamps are out of scope for the primary analysis.

## Pre-registered hypotheses and estimands

All recommendation estimands use eligible users and the specified candidate policy below. Let `NDCG_m,u@10` be binary-relevance NDCG for method `m` and user `u`; a macro mean weights each eligible user once.

- **H1 (homophily):** users assigned to the same friendship community have higher training-only preference similarity than activity-matched users in different communities. The estimand is the mean, over eligible anchor users, of `Jaccard(anchor, within-community peer) - Jaccard(anchor, cross-community peer)` under the sampling procedure below. Positive values support descriptive homophily; they do not establish a friendship effect.
- **H2 (recommendation):** the selected MF + community hybrid improves over selected MF. The primary estimand is `mean_u[NDCG_hybrid,u@10 - NDCG_MF,u@10]` on the single fixed test set. Its paired bootstrap confidence interval is the inferential summary. A positive point estimate with a 95% interval wholly above zero is the predeclared criterion for evidence consistent with improvement.
- **H3 (sparse observed history):** any H2 gain is larger for sparse-history users. The estimands are the H2 difference restricted to each subgroup and, explicitly, `mean(Hybrid - MF | sparse) - mean(Hybrid - MF | non-sparse)`, using the same metric-eligible users and fixed labels in the phase. Sparse status is based only on the training history, so this is not a true new-user cold-start claim.

H2 is the sole primary confirmatory comparison under the approved protocol. H1 and H3 are supporting analyses; global and community-only comparisons, candidate-universe sensitivity results, and shuffled-community results are supporting diagnostics, not substitutes for H2. No multiplicity adjustment is required for that labeling unless the reviewer elects to make H1 or H3 confirmatory.

## Units, deterministic splitting, and eligibility

The indivisible unit is one observed `(user_id, artist_id)` pair with its original positive `weight`; a pair never appears in more than one partition. The fixed split seed is **5101**. For each user with `n` pairs:

1. Sort that user's pairs by `artist_id`. Rank them by ascending SHA-256 digest of the ASCII string `5101|<user_id>|<artist_id>`; break the practically impossible digest tie with ascending `artist_id`.
2. A user is **evaluation-eligible** iff `n >= 7`. Allocate `(train, validation, test)` counts by the largest-remainder apportionment of `(0.70n, 0.15n, 0.15n)`: take floors, then give remaining pairs to largest fractional remainders, breaking ties in the fixed order train, validation, test. Thus every eligible user has at least one pair in each split and at least five training pairs.
3. Assign ranked pairs in order to train, then validation, then test using those counts. The original count is retained unchanged.

Users with fewer than seven observed pairs receive no held-out pairs: all their interactions are marked `train_only_low_history`. They may contribute only to quantities fitted from training data (such as global popularity, MF, and training-only community profiles); they never enter validation/test metrics, tuning, or the sparse subgroup denominator. This preserves all eligible users' training histories and prevents low-history users from silently being evaluated with empty partitions. The report will state the number of excluded users and their interaction count.

No interpretation of the deterministic ordering as time, exposure, or future behavior is permitted.

## Candidate universe and relevance

For any evaluation phase, let `F` be the interactions available for fitting in that phase: split-train plus `train_only_low_history` for validation, and split-train plus split-validation plus `train_only_low_history` after selection for the final test. The **primary candidate universe** is `A_F`, the artists appearing in `F`. An artist is primary-eligible only when it occurs in `F`; an artist absent from `F` is not promoted into the primary universe. For user `u`, every method ranks exactly `A_F \ S_F,u`, where `S_F,u` is that user's artists in `F`. Training-seen artists are therefore never recommended. All methods must use the same user candidate set, including the same deterministic artist-ID tie-break.

The phase relevance set is the user's held-out artists intersected with its candidate set. A held-out artist absent from `A_F` is a **held-out cold item** for the primary policy: it is counted and reported but cannot contribute to a primary metric. A user is metric-eligible in a phase only if the resulting relevance set is non-empty. This makes primary metrics conditional on retrievable held-out relevance; the eligible-user count, users excluded solely for cold relevance, and cold held-out-pair rate must accompany every result.

One predeclared sensitivity analysis uses the full audited artist catalog (`artists.dat`, 17,632 IDs) instead of `A_F`, still excluding `S_F,u` and retaining exactly the same candidates for every method and user. An artist absent from `F` is eligible for this sensitivity candidate set, but it has no learned MF factor and no fit-derived community activity. Consequently, it is not a cold-start result with a learned representation: it is a stress test of candidate-universe expansion and missing-score handling. The sensitivity must report cold-catalog candidate counts and missing-score counts separately, cannot tune on them, and must not be interpreted as evidence that a model solves new-item cold start.

Popularity and community profiles are fitted only from `F`. Validation and test interactions must never construct, smooth, rank, normalize, or otherwise influence fitted popularity/community quantities.

## Implicit-feedback representation and algorithms

Listening `weight` is an aggregate count, not an explicit rating or a negative preference signal. For a fitting interaction define activity `a_ui = log(1 + weight_ui)`, implicit preference `p_ui = 1`, and confidence `c_ui = 1 + alpha * a_ui`; unobserved user--artist pairs have `p_ui = 0`, not a dislike, and baseline confidence 1. The `log1p` transform limits domination by extremely high aggregate counts. `alpha` is tuned only where stated below.

All scores use only fitting data and a stable ascending-`artist_id` tie-break. Score states must be distinguished as follows:

- A **missing learned representation** is an expected absence, such as an MF factor for an artist absent from `F`. It is recorded as `missing_representation`, receives the ranking sentinel negative infinity, and is placed after all rankable candidates; ties among sentinel scores use ascending `artist_id`.
- A **valid zero community probability** is not a missing representation. If `P(i|c)=0`, its log score is negative infinity and it is excluded from that component's finite-score normalization; it remains in the candidate set and is ranked below candidates with finite final scores using the same ID tie-break. The zero may arise only from a defined zero-mass/backoff case, not from an arithmetic failure.
- An **invalid numerical result** is any NaN, positive infinity, negative infinity produced where a finite value is mathematically required, overflow, or undefined normalization (for example, a non-finite MF score or a non-finite z-score for a component that should be finite). It is recorded as `invalid_numerical_result`, is never converted to zero, negative infinity, a mean, or another model's score, and causes that method/configuration/user ranking to be marked failed and omitted from that contrast. A run must not silently substitute invalid model output.

For each user and component, z-normalize only finite raw scores over that user's candidate set: `z_i=(s_i-mean(s))/sd(s)`, with population standard deviation. If there are no finite scores, the component is unavailable and the hybrid ranking fails for that user. If `sd=0`, the component's normalized value is exactly zero for every finite score; this is a valid constant component, not a failure. The hybrid requires finite normalized MF and community components; a missing MF factor, zero-probability log component, or other unavailable component therefore yields a missing final hybrid score for that candidate, not an imputed component. Counts of candidates by missing representation, valid zero probability, and invalid numerical result, plus rankings using the missing sentinel, are reported separately.

- **Global popularity:** score artist `i` by `sum_u a_ui` in `F`; unseen-in-`F` catalog artists have zero score.
- **Implicit-feedback ALS MF:** regularized weighted ALS minimizes the standard implicit objective `sum_ui c_ui(p_ui - x_u^T y_i)^2 + reg*(||X||^2 + ||Y||^2)` over fitting users/items. It uses deterministic initialization and seed 5101. Items absent from `F` have no MF factor and use the missing-representation rule; a factor that exists but produces a non-finite score is an invalid numerical result.
- **Louvain communities:** form one simple undirected friendship graph by canonicalizing pairs as `(min(user_id, friend_id), max(...))`, dropping duplicate directions, and retaining all graph users. Run Louvain with seed 5101 and deterministic sorted node/edge input. Canonicalize each resulting community label by the smallest raw `user_id` in it. Community detection may use the full supplied friendship graph because it contains no held-out listening outcomes; its artist profiles must use `F` only.
- **Community popularity with shrinkage/backoff:** for membership `g(u)`, set `s_ci = sum_{u:g(u)=c} a_ui`, `s_c = sum_i s_ci`, global `q_i = sum_u a_ui / sum_{u,i}a_ui`, and score `P(i|c) = (s_ci + beta*q_i)/(s_c + beta)`. `beta` is pseudo-mass in log-count units. If a community has no fitting activity, use `q_i`; if total fitting activity is unexpectedly zero, use a uniform distribution over phase candidates. Catalog-only items get their smoothed/backoff score.
- **MF + community hybrid:** within each user's candidate set, apply the score-state and finite-value normalization rules above to MF and `log(P(i|g(u)))`; score only candidates with both finite normalized components as `lambda*z_MF + (1-lambda)*z_community`. A missing MF component remains unavailable rather than being imputed; this preserves the declared missing-score accounting.
- **Shuffled-community negative control:** create one seeded permutation of user IDs sorted ascending using seed 5101 and assign each user the community label of its permuted counterpart. This preserves the community-label-size multiset while breaking its association with the original user. Rebuild profiles from `F` with those shuffled labels, and evaluate the community-popularity and hybrid formulations with their selected original-community settings. A material advantage that survives this control weakens a community-specific interpretation.

## Tuning, parameter selection, and final test

Validation is the sole model-selection data. The primary validation metric is macro user NDCG@10 under the primary candidate policy. The global-popularity baseline has no tuned parameter. The following deliberately small grids are proposed:

| Component | Validation-only grid |
|---|---|
| ALS factors | 32, 64 |
| ALS regularization | 0.01, 0.10 |
| ALS confidence `alpha` | 1, 10 |
| ALS iterations | fixed at 20 |
| Community shrinkage `beta` | 10, 100, 1,000 |
| Hybrid `lambda` | 0.25, 0.50, 0.75 |

First select the best of the eight ALS configurations. Next select `beta` for community popularity. Then, holding the selected ALS configuration fixed, select the best of the nine `(beta, lambda)` hybrid configurations. This staged schedule caps fits and is part of the protocol; no retrospective expansion is allowed. Ties in validation NDCG use, in order: fewer factors, larger regularization, smaller `alpha`, smaller `beta`, smaller `lambda`, then lexicographic parameter order. Seed 5101, package versions, hardware-relevant numerical settings, and all grid outcomes must be recorded. For later verification, the recommended established Python 3.11 choices are `implicit==0.7.2` for ALS and `networkx==3.2.1` using its `louvain_communities` implementation for Louvain; exact installed versions and wheel/platform details remain reviewer decisions. These versions are recommendations to verify later, not installed dependencies. Seed 5101 controls the declared initialization, permutation, and sampling operations, but does not imply exact cross-platform bitwise equality: BLAS/OpenMP implementations, thread counts, compiler flags, floating-point reductions, and hardware can change low-order results. Record those settings and compare outputs with an explicitly stated numerical tolerance; if a run is not finite or reproducible within the declared tolerance, mark it failed rather than substituting output.

After approval and parameter selection are locked, refit selected methods on train plus validation data and evaluate the test split once. Test metrics, subgroup results, plots, errors, and candidate sensitivity outcomes cannot trigger parameter changes, feature changes, split changes, or protocol revisions. Any such revision is a new protocol and requires a new split seed/version.

## Evaluation and uncertainty

The primary result is macro-averaged binary-relevance **NDCG@10** over phase metric-eligible users. Secondary outputs are macro Recall@10, catalog coverage@10 (distinct artists recommended in top 10 divided by phase-universe size), candidate-universe size, and all user/pair eligibility counts. NDCG uses `IDCG@10 = sum_{r=1}^{min(10, |relevance|)} 1/log2(r+1)`; Recall divides hits by `|relevance|`.

For each reported method contrast, require the identical metric-eligible user IDs for every successful method in that comparison. Numerical failures must be documented and must not silently alter the evaluation denominator. If a paired comparison cannot be evaluated fairly on one common user set, mark the comparison invalid and report the reason. For valid comparisons, draw 2,000 paired bootstrap resamples of that common user set with replacement using seed 5101. Recompute the macro difference within each resample and report the point estimate and two-sided 95% percentile interval. For H3, independently draw 2,000 user-level resamples with replacement within each fixed sparse and non-sparse group. Preserve each user's paired `Hybrid-MF` difference, calculate both subgroup means and their contrast on every resample, and report the subgroup means, the contrast `mean(Hybrid-MF | sparse) - mean(Hybrid-MF | non-sparse)`, and the two-sided 95% percentile interval. Labels are fixed before validation tuning and are never recomputed. If either group has fewer than 30 metric-eligible users, do not report a bootstrap interval; report the estimates and the insufficiency instead. The user-level bootstrap is a descriptive resampling device for this fixed dataset, not a claim that users or graph-linked observations are independent.

Every result table must also report: raw evaluation-eligible users, metric-eligible users, users with no retrievable held-out relevance, held-out cold pairs, candidate count, rankings with any missing-score substitution, and any failed/omitted runs. A model failure is never silently dropped or replaced with another method's score.

## H1 homophily experiment

Use only training artist sets and only evaluation-eligible users. For every eligible anchor, compute Jaccard similarity of its training artist set with a peer's training artist set. Define fixed training-activity strata as 5--10, 11--20, 21--50, and 51+ distinct training artists. An anchor contributes only if it has at least one other eligible user in its own Louvain community and at least one eligible user in a different community within the same stratum.

For each contributing anchor, rank every eligible within-community candidate peer by the SHA-256 digest of the ASCII string `5101|H1|<anchor_id>|within|<peer_id>`, and rank every eligible cross-community candidate peer by the digest of `5101|H1|<anchor_id>|cross|<peer_id>`. Break hash ties by ascending peer ID. Choose the lowest-ranked candidate separately for the within-community and cross-community comparisons. Record each anchor's paired Jaccard difference and the two selected peer IDs. A peer may recur across anchor comparisons; report its frequency as a peer and do not treat the resulting pair rows as independent. The H1 estimate is the unweighted mean of anchor-level differences, so each anchor contributes once. Its leave-one-user-out jackknife uncertainty is descriptive and approximate: for each user in the eligible H1 graph, remove every row in which that user is the anchor or either selected peer, recompute the mean over the remaining rows, and report the jackknife standard error and a two-sided 95% normal-approximation interval. If fewer than two rows remain for a deletion, omit that deletion and report the number omitted. This procedure acknowledges overlapping users and does not support population-level independence, population-level significance, or causal claims. Report candidate anchor counts by stratum and peer-use frequencies.

## Sparse-history subgroup and safeguards

Define sparse status once, before model selection, among all raw evaluation-eligible users from their split-train distinct-artist counts only: sparse users are those at or below the empirical first quartile, using the nearest-rank order statistic `ceil(0.25*N)` on ascending split-train counts; ties at the cutoff are included. Fix this membership and carry the same label into validation, final refitting, and test. Do not redefine it from train-plus-validation counts during final refitting. Each phase reports its metric-eligible subgroup size. Report the cutoff and the H3 estimands and user-level paired uncertainty specified above. No raw full-history counts, validation/test counts, candidate eligibility, or outcomes may define or revise the subgroup.

The project should fit in one executed notebook and an eight-page report. Parquet files, cached matrices, manifests, and development helpers are intermediate artifacts, not submission substitutes. The final self-contained notebook must reconstruct every required split, mapping, community/profile input, and evaluation ledger from the supplied source and declared configuration without importing unsubmitted local modules; any helper code used during development must be inlined or included in the submitted notebook. Safeguards are: run source/split integrity checks before any fit; cap ALS at eight development fits and hybrid scoring at nine configurations; cache only train-derived matrices; use 32/64 factors and 20 iterations; fail loudly on insufficient memory/time rather than expand compute; and retain parameter/configuration logs. Tags, tag dates, and tag timestamps are excluded from all primary fits and claims.

## Official-instructions compliance checklist

The reviewer independently checked the official DSA5101 Project Instructions at the design level and confirmed the approach is compliant. The official PDF was not available locally for archival comparison. Design requirements already met:

- [x] One HetRec dataset.
- [x] Two distinct lecture algorithm classes: MF (Lecture 5) and Louvain (Lecture 7).
- [x] Coherent combined analysis.
- [x] Baselines and robustness checks.
- [x] Train/validation/test separation.
- [x] Approximately two-week, 8 GB laptop-scale execution scope.
- [x] Reproducibility and citations specified.

Execution/submission requirements pending verification:

- [ ] Eight-page report with the exact required headings.
- [ ] Exactly three final ZIP files.
- [ ] Fresh-kernel notebook execution.

## Dataset, license, and reporting limits

The audit records a bundled non-commercial-use notice, not a standard SPDX license, and unresolved redistribution/derivative-work permissions. The report must preserve provenance and citation information, must not redistribute raw data without separate permission review, and must not claim personal or demographic representativeness. Aggregate listening and static friendship observations may be stale, incomplete, selection-biased, and privacy-sensitive despite the source's stated removal of user names.

## Post-approval implementation checks

1. The approved design uses the `n >= 7` evaluation threshold, split seed 5101, primary candidate policy, and all fixed grids/tie-breaks recorded above.
2. Confirm during implementation that the available Louvain/ALS libraries and their versions can satisfy the deterministic-seed requirement; if not, document an implementation-specific deterministic alternative without changing the approved estimands or split.
3. H2 is the primary confirmatory comparison. H1 and H3 are supporting analyses and are not presented as additional confirmatory tests. Reviewer decision is still required only if a formal multiplicity adjustment is desired for a different confirmatory interpretation; none is required under this labeling.
4. Verify the execution/submission checklist items after notebook execution and report/package assembly.
