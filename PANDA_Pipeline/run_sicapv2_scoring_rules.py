"""Launcher for sicapv2_scoring_rules.py (SICAPv2 Dice under three scoring rules).

Kept separate so that normalization worker processes, which re-import the
launching script on Windows, do not load torch and the pipeline modules.
"""
if __name__ == "__main__":
    import sicapv2_scoring_rules
    sicapv2_scoring_rules.main()
