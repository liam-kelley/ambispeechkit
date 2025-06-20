# Makefile for managing git subtree
# Copy this file to the root of your personal project to manage the ambispeechkit subtree.

# === EXAMPLE USAGE ===
# To add the shared subtree:
#   make add-shared
# To pull updates from the shared subtree:
#   make pull-shared
# To push changes to the shared subtree:
#   make push-shared

# === CONFIGURATION ===
SUBTREE_DIR = ambispeechkit
SUBTREE_REMOTE = https://github.com/liam-kelley/ambispeechkit.git
SUBTREE_BRANCH = main

# === COMMANDS ===

.PHONY: add-shared
add-shared:
	@echo "Adding shared subtree..."
	git subtree add --prefix=$(SUBTREE_DIR) $(SUBTREE_REMOTE) $(SUBTREE_BRANCH) --squash

.PHONY: pull-shared
pull-shared:
	@echo "Pulling latest changes from shared subtree..."
	git subtree pull --prefix=$(SUBTREE_DIR) $(SUBTREE_REMOTE) $(SUBTREE_BRANCH) --squash

.PHONY: push-shared
push-shared:
	@echo "Pushing changes to shared subtree..."
	git subtree push --prefix=$(SUBTREE_DIR) $(SUBTREE_REMOTE) $(SUBTREE_BRANCH)

.PHONY: help
help:
	@echo "Available commands:"
	@echo "  make add-shared   - Add the shared subtree"
	@echo "  make pull-shared  - Pull updates from the shared subtree"
	@echo "  make push-shared  - Push changes to the shared subtree"
