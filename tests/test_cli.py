from vlmforge import __version__
from vlmforge.cli import main


def test_help_runs(capsys):
    assert main([]) == 0
    assert "forge" in capsys.readouterr().out


def test_version(capsys):
    try:
        main(["--version"])
    except SystemExit as e:
        assert e.code == 0
    assert __version__ in capsys.readouterr().out
