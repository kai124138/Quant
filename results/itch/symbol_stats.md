# Converted symbol-day statistics

Source: symbol_stats.csv and symbol_stats_metadata.json; undefined metrics are null.

| date | symbol | raw_messages | messages | dropped_halts | merged_events | valid_book_events | mean_inter_event_gap_seconds | median_inter_event_gap_seconds | mid_changes_per_event | mean_spread_ticks | one_tick_time_share | mean_mid_price_dollars | observed_book_seconds | tick_size_class |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2025-12-08 | AAPL | 0 | 0 | 0 | 0 | 0 | null | null | null | null | null | null | 0.0 | unclassified |
| 2025-12-08 | AMD | 0 | 0 | 0 | 0 | 0 | null | null | null | null | null | null | 0.0 | unclassified |
| 2025-12-08 | AMZN | 0 | 0 | 0 | 0 | 0 | null | null | null | null | null | null | 0.0 | unclassified |
| 2025-12-08 | AVGO | 0 | 0 | 0 | 0 | 0 | null | null | null | null | null | null | 0.0 | unclassified |
| 2025-12-08 | CMCSA | 0 | 0 | 0 | 0 | 0 | null | null | null | null | null | null | 0.0 | unclassified |
| 2025-12-08 | COST | 0 | 0 | 0 | 0 | 0 | null | null | null | null | null | null | 0.0 | unclassified |
| 2025-12-08 | CSCO | 0 | 0 | 0 | 0 | 0 | null | null | null | null | null | null | 0.0 | unclassified |
| 2025-12-08 | GOOGL | 0 | 0 | 0 | 0 | 0 | null | null | null | null | null | null | 0.0 | unclassified |
| 2025-12-08 | INTC | 0 | 0 | 0 | 0 | 0 | null | null | null | null | null | null | 0.0 | unclassified |
| 2025-12-08 | META | 0 | 0 | 0 | 0 | 0 | null | null | null | null | null | null | 0.0 | unclassified |
| 2025-12-08 | MSFT | 0 | 0 | 0 | 0 | 0 | null | null | null | null | null | null | 0.0 | unclassified |
| 2025-12-08 | MU | 0 | 0 | 0 | 0 | 0 | null | null | null | null | null | null | 0.0 | unclassified |
| 2025-12-08 | NFLX | 0 | 0 | 0 | 0 | 0 | null | null | null | null | null | null | 0.0 | unclassified |
| 2025-12-08 | NVDA | 0 | 0 | 0 | 0 | 0 | null | null | null | null | null | null | 0.0 | unclassified |
| 2025-12-08 | PEP | 0 | 0 | 0 | 0 | 0 | null | null | null | null | null | null | 0.0 | unclassified |
| 2025-12-08 | PYPL | 0 | 0 | 0 | 0 | 0 | null | null | null | null | null | null | 0.0 | unclassified |
| 2025-12-08 | QCOM | 0 | 0 | 0 | 0 | 0 | null | null | null | null | null | null | 0.0 | unclassified |
| 2025-12-08 | SBUX | 0 | 0 | 0 | 0 | 0 | null | null | null | null | null | null | 0.0 | unclassified |
| 2025-12-08 | TSLA | 0 | 0 | 0 | 0 | 0 | null | null | null | null | null | null | 0.0 | unclassified |
| 2025-12-08 | WBD | 0 | 0 | 0 | 0 | 0 | null | null | null | null | null | null | 0.0 | unclassified |

## Assumptions

- The research session trim in the metadata applies before event counts and gap summaries.
- Equal timestamps merge using the final book and the Gate 0 loader drops halt rows.
- Tick size is the fixed one-cent convention recorded in the metadata; classes use event-mean spread.
- Missing, locked and crossed books do not contribute to spread or mid averages or time weights.
- Mid changes count adjacent valid event pairs divided by all trimmed events; the first has no predecessor.
- Time shares use observed intervals, including a carried book at the trim boundary, without extending beyond coverage.
- No random sampling is used; the recorded seed is null.
