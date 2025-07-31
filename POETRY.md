# Add a dependency (like npm install package)
poetry add pandas

# Add a dev dependency
poetry add --group dev pytest

# Remove a dependency
poetry remove pandas

# Install all dependencies
poetry install

# Run your script
poetry run python main.py

# Show installed packages
poetry show