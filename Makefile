.PHONY: install train app docker
install:
	uv sync

train:
	uv run python train_all.py

app:
	uv run streamlit run app.py

docker:
	docker build -t deep-learning-mini-suite .
