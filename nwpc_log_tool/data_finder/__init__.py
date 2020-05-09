from pathlib import Path

from nwpc_data.data_finder import find_local_file as _find_local_file


def find_local_file(
        data_type,
        config_dir: str or Path or None = None,
        **kwargs,
) -> Path or None:
    """
    Find files using ``nwpc_data.data_finder.find_local_file`` with config files under conf directory.

    Parameters
    ----------
    data_type: str
        data type, such as "grapes_gfs_gmf/log/fcst_long_std_out"
    config_dir: str or Path or None
        default is None, using conf directory.
    kwargs:
        other parameters using by ``nwpc_data.data_finder.find_local_file``

    Returns
    -------
    Path or None:
        data file path if found or None if not.

    Examples
    --------
    Find std out file for GRAPES GFS GMF.

    >>> find_local_file(
    ...     "grapes_gfs_gmf/log/fcst_long_std_out",
    ...     config_dir="2020050500",
    ... )
    PosixPath('/g1/COMMONDATA/OPER/NWPC/GRAPES_GFS_GMF/Log/2020050421/std.out_fcst_2020050500')

    """
    if config_dir is None:
        config_dir = _get_default_local_config_path()
    return _find_local_file(
        data_type,
        config_dir=config_dir,
        **kwargs,
    )


def _get_default_local_config_path():
    return Path(Path(__file__).parent, "conf").absolute()
