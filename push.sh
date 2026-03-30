#!/bin/bash
# Simple Git push script

# 1️⃣ Optional: check for commit message
if [ -z "$1" ]; then
    echo "Usage: ./push.sh \"Commit message here\""
    exit 1
fi

COMMIT_MSG="$1"

# 2️⃣ Stage all changes
git add .

# 3️⃣ Commit
git commit -m "$COMMIT_MSG"

# 4️⃣ Push to current branch
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
git push origin "$CURRENT_BRANCH"

echo "✅ Pushed changes to branch '$CURRENT_BRANCH' on GitHub" 
