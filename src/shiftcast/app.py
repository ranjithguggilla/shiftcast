"""Launch the Gradio planner: `python -m shiftcast.app`."""

from .service import build_app

if __name__ == "__main__":
    build_app().launch()
