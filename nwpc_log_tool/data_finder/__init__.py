from pathlib import Path

from nwpc_data.data_finder import find_local_file as _find_local_file


def find_local_file(
        data_type,
        config_dir: str or Path or None = None,
        **kwargs,
) -> Path or None:
    if config_dir is None:
        config_dir = _get_default_local_config_path()
    return _find_local_file(
        data_type,
        config_dir=config_dir,
        **kwargs,
    )


def _get_default_local_config_path():
    return Path(Path(__file__).parent, "conf").absolute()
