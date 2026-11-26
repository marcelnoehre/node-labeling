import argparse

from typing import Any, Dict, Optional

from node_labeling.utils.config import Config

def label(graphml: str, cfg: Optional[Config] = None, export: bool = False) -> Optional[Dict[int, Any]]:
    '''
    Run the node labeling pipeline end to end, in the version chosen by `cfg.version`.

    Parameters
    ----------
    graphml : str
        Path to a .graphml file. Each node needs `x`/`y` position data, and
        optionally `general` data as a plain string and `extent`/`intent`
        label data (`;`-separated names).
    cfg : Config, optional
        Configuration overrides. Defaults to `Config()`.
    export : bool
        If True, write the final layout to a PDF in `figs/` and return None.
        If False, skip the export and return the computed label placements instead.

    Returns
    -------
    labels : Optional[Dict[int, Any]]
        The final placed label for each label id, as the label model of the chosen
        version, or None if export is True.
    '''
    cfg = cfg or Config()
    if cfg.version == 1:
        from node_labeling.v1.pipeline import label as run
    elif cfg.version == 2:
        from node_labeling.v2.pipeline import label as run
    else:
        raise ValueError(f'unknown pipeline version {cfg.version!r}, expected 1 or 2')
    return run(graphml, cfg, export)


def main():
    parser = argparse.ArgumentParser(description='Node Labeling: two-phase labeling of line diagrams of ordered sets')
    parser.add_argument(
        '--export', action='store_true',
        help='Write the final layout to a PDF in figs/ instead of printing the label placements'
    )
    parser.add_argument(
        '--version', type=int, choices=(1, 2), default=Config.version,
        help='Pipeline version (default: %(default)s)'
    )
    args = parser.parse_args()

    graphml = input('Path to .graphml file: ')
    labels = label(graphml, Config(version=args.version), export=args.export)
    if labels is not None:
        print(labels)


if __name__ == '__main__':
    main()
