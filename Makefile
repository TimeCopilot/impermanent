MODAL = uv run modal run -m

## GH Archive Extract

.PHONY: validate-extract
validate-extract: # Validate raw events
	$(MODAL) src.data.gh_archive.extract.modal_app::validate_raw_events_from_start_to_current --start-date 2023-01-01-00

## GH Archive Transform

.PHONY: validate-transform
validate-transform: # Validate transformed hours
	$(MODAL) src.data.gh_archive.transform.modal_app::validate_transformed_from_start_to_current --start-date 2023-01-01-00

## GH Archive Aggregate

AGG_START_daily   := 2023-01-02-00
AGG_START_weekly  := 2023-01-09-01
AGG_START_monthly := 2023-02-01-00
AGG_FREQ_daily    := D
AGG_FREQ_weekly   := W-SUN
AGG_FREQ_monthly  := MS
AGG_FREQUENCIES   := daily weekly monthly

.PHONY: $(addprefix validate-aggregate-,$(AGG_FREQUENCIES))
$(addprefix validate-aggregate-,$(AGG_FREQUENCIES)): validate-aggregate-%:
	$(MODAL) src.data.gh_archive.aggregate.modal_app::validate_aggregated_from_start_to_current --start-date $(AGG_START_$*) --freq $(AGG_FREQ_$*)

## GH Archive Forecast

FC_START_daily    := 2026-01-04
FC_START_weekly   := 2026-01-04
FC_START_monthly  := 2025-10-01
FC_FREQUENCIES    := daily weekly monthly

.PHONY: $(addprefix validate-forecast-,$(FC_FREQUENCIES))
$(addprefix validate-forecast-,$(FC_FREQUENCIES)): validate-forecast-%:
	$(MODAL) src.forecast.gh_archive.modal_app::validate_forecasts_from_start_to_current --start-date $(FC_START_$*) --frequency $*

## GH Archive Evaluate

EV_START_daily    := 2026-01-04
EV_START_weekly   := 2026-01-04
EV_START_monthly  := 2025-10-01
EV_FREQUENCIES    := daily weekly monthly

.PHONY: $(addprefix validate-evaluate-,$(EV_FREQUENCIES))
$(addprefix validate-evaluate-,$(EV_FREQUENCIES)): validate-evaluate-%:
	$(MODAL) src.evaluation.gh_archive.modal_app::validate_evaluations_from_start_to_current --start-date $(EV_START_$*) --frequency $*

.PHONY: leaderboard
leaderboard: # Build leaderboard parquet from all evaluation parquets
	$(MODAL) src.evaluation.gh_archive.modal_app::build_leaderboard
