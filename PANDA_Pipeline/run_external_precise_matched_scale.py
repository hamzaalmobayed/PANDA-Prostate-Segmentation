"""Launcher for precise_matched_scale.py (PRECISE at the PANDA pixel size).

Kept separate so that normalization worker processes, which re-import the
launching script on Windows, do not load torch and the pipeline modules.
"""
if __name__ == "__main__":
    import precise_matched_scale
    precise_matched_scale.main()
