#!/bin/bash

echo "🚀 Publishing NNX Neuroplasticity to GitHub"
echo "=============================================="

# Check if remote is already set
if git remote get-url origin >/dev/null 2>&1; then
    echo "✅ Remote origin already configured"
    echo "Current remote URL: $(git remote get-url origin)"
else
    echo "❌ No remote origin configured"
    echo ""
    echo "📝 To publish to GitHub:"
    echo "1. Go to https://github.com/new"
    echo "2. Create repository: nnx-neuroplasticity"
    echo "3. Run: git remote add origin https://github.com/YOUR_USERNAME/nnx-neuroplasticity.git"
    echo "4. Run: git push -u origin main"
    echo ""
    echo "🔗 Repository URL will be: https://github.com/YOUR_USERNAME/nnx-neuroplasticity"
fi

echo ""
echo "📊 Repository Status:"
echo "   - Commits: $(git log --oneline | wc -l)"
echo "   - Files: $(git ls-files | wc -l)"
echo "   - Branch: $(git branch --show-current)"

echo ""
echo "📁 Files included:"
git ls-files | head -10
if [ $(git ls-files | wc -l) -gt 10 ]; then
    echo "   ... and $(($(git ls-files | wc -l) - 10)) more files"
fi 