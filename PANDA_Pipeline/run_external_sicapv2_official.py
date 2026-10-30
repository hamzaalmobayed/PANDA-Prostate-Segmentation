"""Launcher for sicapv2_official.py (SICAPv2 official release).

Kept separate so that normalization worker processes, which re-import the
launching script on Windows, do not load torch and the pipeline modules.
"""
if __name__ == "__main__":
    import sicapv2_official
    sicapv2_official.main()
