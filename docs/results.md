# Evaluation results

- **Generated:** 2026-11-05
- **Model:** claude-haiku-5-5 on every node
- **Matchups:** eval/matchups_v1.json (40 games, 2026 weeks 4–7, after every model's training cutoff)
- **Data snapshot:** data/snapshots/nfl_2026-11-02.duckdb (sha256 75d205650945)

## Main comparison

Percentages show a 95% bootstrap interval over reports.

| Metric | Full graph | No fact-checker | Single agent |
|---|---|---|---|
| Reports | 120 | 120 | 120 |
| Draft claims | 10,611 | 10,647 | 5,089 |
| Draft claim accuracy | 99.0% (98.7–99.2) | 99.1% (98.9–99.3) | 99.3% (99.0–99.6) |
| Final claim accuracy | 100.0% (100.0–100.0) | 99.5% (99.3–99.7) | 99.5% (99.3–99.7) |
| Draft wording correct | 97.7% | 97.2% | 98.9% |
| Final wording correct | 100.0% | 96.9% | 98.9% |
| Claims written with no source | 0 | 0 | 0 |
| Unsupported numbers per draft | 1.3 | 1.5 | 0.7 |
| Report numbers per report | 65.9 | 69.0 | 42.2 |
| Report numbers unsupported | 0.1% | 0.3% | 1.6% |
| Report numbers mismatched | 0.0% | 0.1% | 0.0% |
| Claims corrected by revision | 244 | 0 | 0 |
| Claims dropped | 120 | 0 | 0 |
| Independent sample: tool agrees | 100.0% | 100.0% | 100.0% |
| Independent sample: stated value agrees | 99.5% | 99.8% | 99.3% |
| Seeded errors caught | 99.7% | 99.2% | 99.7% |
| Checker false positives (seeded) | 1 | 2 | 1 |
| Tool error rate | 1.7% | 2.3% | 5.8% |
| Output tokens per report | 48,797 | 33,243 | 20,380 |
| Cost per report | $0.031 | $0.021 | $0.013 |

## Seeded errors caught, by kind

| Kind | Full graph | No fact-checker | Single agent |
|---|---|---|---|
| unlinked_number | 100.0% | 100.0% | 100.0% |
| wrong_split | 98.6% | 96.9% | 99.4% |
| wrong_team | 100.0% | 100.0% | 99.4% |
| wrong_value | 100.0% | 100.0% | 100.0% |

## Draft claim accuracy by repeat

| Config | r1 | r2 | r3 |
|---|---|---|---|
| full | 99.1% | 98.9% | 98.8% |
| no_checker | 99.3% | 98.9% | 99.1% |
| single | 99.2% | 99.3% | 99.4% |

## Forward check

For each team-level number a final report cites: is the season-to-date value closer to the same metric in the game itself than the league average through the same week? Each (game, metric, split) counts once.

| Config | Claims | Report closer than league average |
|---|---|---|
| Full graph | 1,214 | 46.8% |
| No fact-checker | 1,249 | 46.2% |
| Single agent | 1,049 | 48.3% |

By metric (Full graph):

| Metric | Claims | Report closer | MAE report | MAE league |
|---|---|---|---|---|
| average_depth_of_target | 39 | 41.0% | 2.166 | 1.762 |
| epa_per_play | 290 | 47.2% | 0.206 | 0.201 |
| explosive_play_rate | 194 | 47.4% | 4.544 | 4.376 |
| pass_rate | 116 | 44.8% | 10.254 | 9.622 |
| pressure_proxy_rate | 106 | 41.5% | 5.028 | 4.375 |
| run_direction_rate | 107 | 58.9% | 11.383 | 12.210 |
| sack_rate | 79 | 45.6% | 3.746 | 3.449 |
| success_rate | 157 | 45.2% | 8.415 | 7.925 |
| third_down_conversion_rate | 126 | 45.2% | 10.977 | 10.120 |


## Notes

- **Single-agent crashes:** 8 of the single agent's first 120 attempts (6.7%) crashed. Each wrote a claim with no source, and claim validation failed. The multi-agent graph had none in 240 attempts. The fix sets such claims aside (their numbers count as unsupported), and the 8 reports were regenerated: none had an unsourced claim. The single agent's real reliability is therefore lower than the "0" row suggests.
- **Seeded-error rows describe the checker, not the config.** Each config's drafts were seeded and run through the same deterministic checker. In the pipeline, the checker only runs in the full graph.
- **How accuracy is judged.** The project's checker re-runs each cited tool call. As an independent check, about 2,840 claims were re-derived with separate code. The tools agreed 100%, and the stated values agreed 99.3–99.8%.
- **Forward check is about the data, not the configs.** Season-to-date numbers through weeks 3–6 are no closer to the next game than the league average (46–48% "closer" in every config). The one exception is run direction. With 3–6 games of data, team splits regress toward the mean.
- **Not measured yet:** usefulness. The blind hand-grading of 30 reports is pending.
- **Scope:** one model (Claude Haiku 5.5), 40 matchups, 3 repeats per config. The MLflow store also holds an earlier 2-matchup pilot scoring.
