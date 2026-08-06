.PHONY: install run deploy logs restart stop status

install:
	python3 -m venv .venv
	.venv/bin/pip install -e .

run:
	@if sudo systemctl is-active --quiet ai-character; then \
		echo "Service ai-character is running, stopping it..."; \
		sudo systemctl stop ai-character; \
	fi
	.venv/bin/python -m src.main

deploy:
	git push deploy master

logs:
	sudo journalctl -u ai-character -f

update:
	git pull origin main
	.venv/bin/pip install -e .
	sudo systemctl restart ai-character

restart:
	sudo systemctl restart ai-character

stop:
	sudo systemctl stop ai-character

status:
	sudo systemctl status ai-character

setup-server:
	sudo apt update && sudo apt install -y python3 python3-venv python3-dev git
	sudo useradd -r -s /bin/false bot || true
	sudo mkdir -p /opt/ai_character
	sudo chown bot:bot /opt/ai_character

setup-service:
	sudo cp deploy/ai-character.service /etc/systemd/system/
	sudo systemctl daemon-reload
	sudo systemctl enable ai-character
	sudo systemctl start ai-character
