from nitorch_cli.parser import ParseError
from .main import info
from .parser import parse, help
import sys


def entrypoint(args=None):
    f"""Command-line interface for `niinfo`
    
    {help}
    
    """

    # Exceptions are dealt with here
    try:
        _entrypoint(args)
    except ParseError as e:
        print(help)
        print(f'[ERROR] {str(e)}', file=sys.stderr)
    except Exception as e:
        print(f'[ERROR] {str(e)}', file=sys.stderr)


def _entrypoint(args):
    """Command-line interface for `niinfo` without exception handling"""
    args = args or sys.argv[1:]

    options = parse(args)
    if not options:
        return

    for i, file in enumerate(options.files):
        info(file, meta=options.meta, stat=options.stat)
        if i < len(options.files) - 1:
            print('-' * 79)

