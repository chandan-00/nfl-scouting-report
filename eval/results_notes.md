## Notes

- **Single-agent crashes:** 8 of the single agent's first 120 attempts (6.7%) crashed. Each wrote a claim with no source, and claim validation failed. The multi-agent graph had none in 240 attempts. The fix sets such claims aside (their numbers count as unsupported), and the 8 reports were regenerated: none had an unsourced claim. The single agent's real reliability is therefore lower than the "0" row suggests.
- **Seeded-error rows describe the checker, not the config.** Each config's drafts were seeded and run through the same deterministic checker. In the pipeline, the checker only runs in the full graph.
- **How accuracy is judged.** The project's checker re-runs each cited tool call. As an independent check, about 2,840 claims were re-derived with separate code. The tools agreed 100%, and the stated values agreed 99.3–99.8%.
- **Forward check is about the data, not the configs.** Season-to-date numbers through weeks 3–6 are no closer to the next game than the league average (46–48% "closer" in every config). The one exception is run direction. With 3–6 games of data, team splits regress toward the mean.
- **Not measured yet:** usefulness. The blind hand-grading of 30 reports is pending.
- **Scope:** one model (Claude Haiku 5.5), 40 matchups, 3 repeats per config. The MLflow store also holds an earlier 2-matchup pilot scoring.
