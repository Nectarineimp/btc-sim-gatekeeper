#!/bin/bash
poetry run python src/simulate.py --data ~/projects/data/btc_daily_price.csv --output output/gatekeeper_forecast.csv --iterations 10000
