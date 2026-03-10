"""Entry point: python -m aiqyn"""
import sys

def main() -> None:
    # If no CLI args → launch GUI
    if len(sys.argv) == 1:
        from aiqyn.ui.app import run_app
        run_app()
    else:
        from aiqyn.cli.main import app
        app()

if __name__ == "__main__":
    main()
