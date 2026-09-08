.PHONY: help install pipeline dashboard reports notebook test verify all clean

help:
	@echo "install    install dependencies"
	@echo "pipeline   data -> models -> evaluation -> expected loss -> figures"
	@echo "reports    regenerate README, executive summary and risk memo"
	@echo "dashboard  rebuild the static HTML dashboard"
	@echo "notebook   regenerate and execute the technical notebook"
	@echo "app        launch the interactive Streamlit dashboard"
	@echo "test       run the regression tests on the risk maths"
	@echo "verify     recompute every headline number independently"
	@echo "all        pipeline + reports + dashboard + notebook + test"

install:
	pip install -r requirements.txt

pipeline:
	python -m src.run_pipeline

reports:
	python -m src.build_reports

dashboard:
	python -m src.build_static_dashboard

notebook:
	python tools/make_notebook.py

app:
	streamlit run dashboard/app.py

test:
	pytest -q

verify:
	python tools/verify_results.py

all: pipeline reports dashboard notebook test verify

clean:
	rm -rf data/processed/* outputs/models/* __pycache__ src/__pycache__ .pytest_cache
