.PHONY: install clean rebuild logs

install-d:
	@echo "Setting up SocialMetrics AI..."
	@echo "Creating .env file if it doesn't exist"
	@[ -f .env ] || cp .env.example .env
	@echo "Setting up Docker volume directories"
	@if [ ! -d "./.docker/mysql" ]; then \
		echo "Creating .docker/mysql directory"; \
		mkdir -p ./.docker/mysql; \
	fi
	@if [ ! -d "./app/data" ]; then \
			echo "Creating app/data directory"; \
			mkdir -p ./app/data; \
		fi
	@echo "Ensuring scripts are executable"
	@chmod +x wait-for-db.sh init-cron.sh
	@echo "Building and starting Docker containers..."
	make build -d
	@echo "Waiting for MySQL to be ready..."
	@sleep 20
	@echo "Initializing database with sample data..."
	docker-compose exec app python setup.py db || echo "Failed to initialize database, will retry later"
	@echo "Training initial model..."
	docker-compose exec app python setup.py model || echo "Failed to train model, will retry later"
	@echo "Installation complete! Access the application at http://localhost:5001"

install:
	@echo "Setting up SocialMetrics AI..."
	@echo "Creating .env file if it doesn't exist"
	@[ -f .env ] || cp .env.example .env
	@echo "Setting up Docker volume directories"
	@if [ ! -d "./.docker/mysql" ]; then \
		echo "Creating .docker/mysql directory"; \
		mkdir -p ./.docker/mysql; \
	fi
	@if [ ! -d "./app/data" ]; then \
    		echo "Creating app/data directory"; \
    		mkdir -p ./app/data; \
    	fi
	@echo "Ensuring scripts are executable"
	@chmod +x wait-for-db.sh init-cron.sh
	@echo "Building and starting Docker containers..."
	make build
	@echo "Waiting for MySQL to be ready..."
	@sleep 20
	@echo "Initializing database with sample data..."
	docker-compose exec app python setup.py db || echo "Failed to initialize database, will retry later"
	@echo "Training initial model..."
	docker-compose exec app python setup.py model || echo "Failed to train model, will retry later"
	@echo "Installation complete! Access the application at http://localhost:5001"

build-d:
	docker-compose up -d --build

build:
	docker-compose up --build

clean:
	@echo "Stopping containers and removing volumes..."
	docker-compose down -v
	@echo "Cleaning up temporary files..."
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

rebuild:
	@echo "Rebuilding the application..."
	docker-compose down
	docker-compose up -d --build

logs:
	@echo "Showing application logs..."
	docker-compose logs -f

db-logs:
	@echo "Showing database logs..."
	docker-compose logs -f db

app-logs:
	@echo "Showing app logs..."
	docker-compose logs -f app

cron-logs:
	@echo "Showing cron logs..."
	docker-compose logs -f cron

init-db:
	@echo "Initializing database with sample data..."
	docker-compose exec app python setup.py db

train-model:
	@echo "Training the model..."
	docker-compose exec app python setup.py model