"""The web folder is served at /extensions/<pack>/, which constrains it."""

from pathlib import Path

WEB = Path(__file__).resolve().parents[1] / "web"


def test_scripts_sit_at_the_served_root():
    # a script in web/<sub>/ is served at /extensions/<pack>/<sub>/, where
    # "../../scripts/app.js" resolves to /extensions/scripts/app.js (404) and
    # the module never runs
    for script in WEB.rglob("*.js"):
        assert script.parent == WEB, f"{script} must live in web/, not a subfolder"


def test_scripts_import_the_app_from_the_right_depth():
    for script in WEB.rglob("*.js"):
        source = script.read_text()
        if "scripts/app.js" in source:
            assert '"../../scripts/app.js"' in source, script


def test_the_pack_exports_a_web_directory():
    root_init = (WEB.parent / "__init__.py").read_text()
    assert 'WEB_DIRECTORY = "./web"' in root_init
