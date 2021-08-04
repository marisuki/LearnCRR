init:
	pip3 install -r requirements.txt

test:
	python3 test/test_birdmap.py
	python3 test/test_abalone.py

.PHONY: init test