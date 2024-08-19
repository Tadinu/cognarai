git fsck --unreachable --no-reflog
git reflog expire --expire=now --all 
git gc --prune=now
