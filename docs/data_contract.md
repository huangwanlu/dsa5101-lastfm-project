# DSA5101 HetRec Last.fm data contract

**Status: FROZEN — lastfm-v1.** This is a design freeze, not proof of successful implementation, numerical execution, or library compatibility. It authorizes no extraction, split generation, fitting, or evaluation until the implementation validation gates are passed.

**Protocol identifier:** `lastfm-v1`

## Status conventions

- **Confirmed** facts are drawn from the local P0.5 audit of the supplied archive.
- **Proposed** items are the reproducibility contract to adopt after review.
- **Unresolved** items require reviewer approval or information not present locally.

## Source provenance and loading contract

**Confirmed source:** `data/raw/hetrec2011-lastfm-2k.zip`, 2,589,075 bytes, SHA-256 `6738f48195667ff03caaab4d32ca9a3133d8cc026b7c3cdaf6ce1010e913c59c`. It is HetRec 2011 Last.fm 2K, version 1.0 (May 2011). Read members directly from the ZIP; do not treat an extracted copy as authoritative. ZIP CRC validation must pass before parsing.

| ZIP member | Header / schema | Encoding | Primary use |
|---|---|---|---|
| `user_artists.dat` | `userID`, `artistID`, `weight` | ASCII-compatible UTF-8 | Required interactions |
| `user_friends.dat` | `userID`, `friendID` | ASCII-compatible UTF-8 | Required friendship graph |
| `artists.dat` | `id`, `name`, `url`, `pictureURL` | UTF-8 | Required artist catalog/candidate sensitivity |
| `tags.dat` | `tagID`, `tagValue` | Windows-1252 | Audit only; excluded from primary analyses |
| `user_taggedartists.dat` | `userID`, `artistID`, `tagID`, `day`, `month`, `year` | ASCII-compatible UTF-8 | Audit only; excluded |
| `user_taggedartists-timestamps.dat` | `userID`, `artistID`, `tagID`, `timestamp` | ASCII-compatible UTF-8 | Audit only; excluded |
| `readme.txt` | prose | Windows-1252 | Provenance/license/citation only |

The parser must decode each member using the stated encoding, delimiter TAB, exactly the stated header, and no lossy replacement decoding. File/member hashes and the archive checksum must be placed in a run manifest. The archive's bundled notice permits non-commercial use but does not state a standard open-data license or clear redistribution terms; raw data and substantial extracts must remain untracked and unredistributed pending permission review.

## Canonical logical schemas

All identifiers and counts are parsed as signed 64-bit integers. IDs are positive; fields marked string are decoded Unicode strings. Input names remain camel-case only at raw-file boundaries; derived fields use snake_case.

| Logical table | Fields and types | Key and required invariants |
|---|---|---|
| `interactions_raw` | `user_id:int64`, `artist_id:int64`, `weight:int64` | Primary key `(user_id, artist_id)`; all values positive; one row per observed pair; every `artist_id` exists in `artists_raw` |
| `friend_edges_raw` | `user_id:int64`, `friend_id:int64` | Directed key `(user_id, friend_id)`; positive IDs; no self-loop; each endpoint is an interaction user; reciprocal rows are expected |
| `friend_edges_undirected` | `user_low_id:int64`, `user_high_id:int64` | Key `(user_low_id, user_high_id)` where low < high; exact canonicalization of directed reciprocal input; no duplicates or loops |
| `artists_raw` | `artist_id:int64`, `name:string`, `url:string`, `picture_url:string nullable` | Primary key `artist_id`; ID positive/unique; name and URL nonempty; blank picture URL is preserved as null/empty per parser convention and is not a rejection |
| `tags_raw` | `tag_id:int64`, `tag_value:string` | Primary key `tag_id`; audit only |
| `tag_assignments_date_raw` | `user_id:int64`, `artist_id:int64`, `tag_id:int64`, `day:int32`, `month:int32`, `year:int32` | Key `(user_id, artist_id, tag_id)`; audit only |
| `tag_assignments_ts_raw` | `user_id:int64`, `artist_id:int64`, `tag_id:int64`, `timestamp_ms:int64` | Same key/order as date assignment table; audit only |

**Confirmed audit invariants:** 92,834 unique interaction pairs; 1,892 interaction users; 17,632 interaction artists, all catalogued; positive weights 1--352,698. The friendship input has 25,434 unique reciprocal directed rows and 12,717 unique undirected edges. These counts are source-version checks, not assumptions for alternative source versions.

## Identifier, ordering, and numeric conventions

**Proposed:** preserve raw numeric IDs as the external keys everywhere. For arrays, create versioned mapping artifacts with IDs sorted ascending and a zero-based contiguous `index`:

- `user_index`: every interaction user, sorted `user_id` ascending;
- `artist_index`: every catalog artist, sorted `artist_id` ascending;
- `fit_artist_index`: artists present in a named fitting partition, sorted `artist_id` ascending.

An implementation may use internal indices only through the matching artifact/version. It must not derive indices from ingestion order, hash-map iteration, or community-library labels. All row-oriented artifacts sort by `user_id`, then `artist_id` unless this contract says otherwise. Floating values are IEEE-754 float64 during calculation; serialized metric values record sufficient precision for reproduction. Raw weights are never overwritten by confidence or transformed activity values.

Community labels are not trusted from Louvain as stable IDs. After detection, each membership label is canonically replaced by the smallest raw `user_id` in that community; memberships are stored sorted by `user_id`.

## Proposed derived artifacts (not yet created)

All generated artifacts are ignored under `data/processed/` and must live below `data/processed/lastfm-v1/`, where the directory identifier changes for any approved split/protocol revision. Each artifact requires a sidecar `manifest.json` recording archive checksum, code revision, configuration, seed, package versions, creation UTC timestamp, row count, schema, and file SHA-256.

| Artifact | Required fields | Integrity rules |
|---|---|---|
| `interaction_splits.parquet` | `user_id:int64`, `artist_id:int64`, `weight:int64`, `split:string` | Exact one-to-one copy of every raw interaction; `split` in `train`, `validation`, `test`, `train_only_low_history`; primary key retained; sorted as specified |
| `split_manifest.json` | source checksum, seed `5101`, eligibility threshold `7`, allocation method, hash-ranking specification, per-split user/pair counts | Recomputable from raw interactions; digest/row counts must match table |
| `user_index.parquet` | `user_id:int64`, `index:int32` | One row per interaction user; sorted IDs; indices exactly `0..n-1` |
| `artist_index.parquet` | `artist_id:int64`, `index:int32` | One row per catalog artist; sorted IDs; indices exactly `0..n-1` |
| `fit_artist_index_<phase>.parquet` | `artist_id:int64`, `index:int32` | Exact sorted distinct artists in phase fitting set, for `validation` or `test` |
| `community_membership.parquet` | `user_id:int64`, `community_id:int64` | One row per graph user; canonical labels; Louvain seed/version/modularity in manifest |
| `evaluation_ledger_<phase>.parquet` | `user_id:int64`, `raw_evaluation_eligible:bool`, `metric_eligible:bool`, `split_train_history_count:int32`, `fit_history_count:int32`, `heldout_count:int32`, `retrievable_heldout_count:int32`, `cold_heldout_count:int32`, `candidate_count:int32`, `sparse:bool` | One row per interaction user; `sparse` is the once-frozen split-train-count label; phase definitions exactly match protocol; never contains a test-derived fitting feature |
| `run_manifest.json` | artifact checksums, method/configuration, seeds, phase, candidate policy, dependency versions, missing-score counts | Immutable record for each fit/evaluation run |

`validation` fitting data is rows with `split in {train, train_only_low_history}`. `test` fitting data after protocol freeze is rows with `split in {train, validation, train_only_low_history}`. The sparse label is the sole exception to phase-derived quantities: calculate it once from split-train distinct-artist counts among all raw evaluation-eligible users, freeze it before validation tuning, and carry it unchanged through validation, final refitting, and test reporting. Final-refit data must not define, update, or revise sparse labels. Only the stated fitting set can create profiles, transformations, MF factors, normalization statistics, or the phase candidate artist universe. Test rows may be read only to calculate already-frozen test metrics and ledgers after final ranking.

## Split and partition invariants

The exact proposed splitter is defined in the protocol: seed 5101; per-user SHA-256 ranking of `5101|user_id|artist_id`; largest-remainder 70/15/15 allocation with train/validation/test tie priority; `n >= 7` for evaluation eligibility. The contract additionally requires:

1. Every raw interaction appears in exactly one split row with identical IDs and weight; no synthesized zero interactions or duplicate pairs.
2. For each user with `n >= 7`, split sizes equal the specified apportionment and each of train/validation/test is nonempty. For `n < 7`, all rows are `train_only_low_history` and none are held out.
3. A user's fitting-history set for a phase is disjoint from that phase's held-out set. User--artist pairs are never divided, and weights never cross partition boundaries.
4. Re-running the splitter on the same archive checksum and configuration produces byte-equivalent sorted rows or a documented semantically equivalent serialization with identical manifest digest.
5. The split manifest must record users/pairs by split, raw evaluation-eligible users, and users excluded for low history.

## Candidate, relevance, and leakage boundaries

For each phase, `candidate_universe` is either the primary distinct artist IDs in the fitting set or, only for the predeclared sensitivity, all `artist_index` IDs. Per-user candidates equal that universe minus their fitting artist IDs. The candidate universe, candidate set, relevance set, and eligible-user ledger must be materialized or reconstructible solely from the split table and phase declaration.

The phase relevance set is held-out artist IDs intersected with that user's candidate set. Primary held-out artists absent from the fit-artist universe are recorded as cold, not silently counted as misses or used for fitting. Every method receives the identical candidate list. For each method comparison, require the identical metric-eligible user IDs across all successful methods; numerical failures are documented and must not silently alter the denominator. If a fair paired comparison cannot be formed, mark it invalid and report why. A missing learned representation (for example, an MF factor absent for a catalog-only artist in the full-catalog sensitivity) is recorded separately from an invalid numerical result and receives only the documented negative-infinity ranking sentinel. A valid zero community probability maps to a negative-infinity log component and is not an arithmetic failure. NaN, unexpected infinity, overflow, or undefined normalization where a finite value is required is `invalid_numerical_result`; it must fail/omit the affected ranking or run and must never be imputed or silently replaced. Finite scores are z-normalized per user with population standard deviation; a zero standard deviation maps finite values to zero. The hybrid requires finite normalized components. All sentinel ties use ascending `artist_id`, and all missing/zero/invalid counts are recorded. Sparse status is calculated once from split-train distinct-artist counts among raw evaluation-eligible users, using the protocol's nearest-rank first-quartile rule; candidate/relevance outcomes and later validation rows cannot change it.

The following are prohibited before final test scoring: reading validation/test rows to calculate popularity, community artist profiles, activity/confidence transforms, MF factors, score normalization, candidate membership, sparse thresholds, hyperparameters, or community-profile shrinkage. Friendship topology may be used for Louvain membership because it is a separate static input, but all community **artist** quantities must be fit-only. Tags, tag assignment dates, and tag timestamps are prohibited from primary features, splits, communities, and outcomes.

## Source anomalies and non-primary tag policy

**Confirmed:** `artists.dat` has 444 blank picture URLs. Tag files are CP1252/ASCII-compatible as listed above. Tag assignment files share 186,479 keys in the same order, but contain 1,538 assignments involving 390 artist IDs absent from `artists.dat`; four timestamps are negative and corresponding dates are 1956--1957. Tag timestamp dates depend on `Europe/Madrid` interpretation, and none are listening events.

**Policy:** retain these observations only for audit/provenance if tag files are parsed. Do not enforce a catalog foreign key on tag-assignment artist IDs; do not use tags or their timestamps in the primary protocol; do not infer listening chronology from them. `artists.dat` is, however, a complete catalog foreign key for `interactions_raw` in this audited archive.

## Validation gates and unresolved items

Before any implementation, a run must validate archive SHA-256/CRC, member names, headers, encodings, schema widths, raw keys, source count checks, mapping bijections, split invariants, fit/held-out disjointness, candidate equality across methods, and the absence of test/validation fields in fitting artifacts. It must also check that raw data is not staged/tracked.

The proposed artifact format/version path and split design are frozen as `lastfm-v1`. The exact installed Louvain/ALS implementations and versions remain implementation checks; the recommended later-verification choices are `implicit==0.7.2` on Python 3.11 for ALS and `networkx==3.2.1` using `louvain_communities` for Louvain; do not install them as part of this documentation pass. Seed 5101 is deterministic control, not a guarantee of cross-platform bitwise identity; record BLAS/OpenMP, thread, compiler, hardware, and wheel details and use a declared numerical tolerance. The final submitted notebook must reconstruct necessary artifacts from the source and configuration without importing unsubmitted local modules; Parquet files and development helpers remain intermediate artifacts. H2 is the primary confirmatory comparison, while H1 and H3 are supporting analyses unless a reviewer explicitly approves a different multiplicity treatment. The reviewer independently checked the official DSA5101 Project Instructions at the design level; execution/submission requirements remain pending verification after implementation. The protocol preserves the report and submission constraints in its checklist.
