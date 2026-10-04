"""Compatibility entry point for author-versus-retrained analysis.

Prefer ``python analysis/compare_author_vs_retrained.py --help``. This wrapper
keeps the former top-level command name while requiring the new explicit paths.
"""

from analysis.compare_author_vs_retrained import main


if __name__ == "__main__":
    main()
